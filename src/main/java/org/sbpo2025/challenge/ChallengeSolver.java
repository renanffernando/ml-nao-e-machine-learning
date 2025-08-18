package org.sbpo2025.challenge;

import org.apache.commons.lang3.time.StopWatch;

import java.io.BufferedReader;
import java.io.EOFException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

import ilog.concert.*; // Classes base (restrições, variáveis, expressões)
import ilog.cplex.*; // O solver CPLEX

public class ChallengeSolver {
    private final long MAX_RUNTIME = 600000; // milliseconds; 10 minutes

    protected List<Map<Integer, Integer>> orders;
    protected List<Map<Integer, Integer>> aisles;
    protected int nItems;
    protected int waveSizeLB;
    protected int waveSizeUB;

    public ChallengeSolver(
            List<Map<Integer, Integer>> orders, List<Map<Integer, Integer>> aisles, int nItems, int waveSizeLB,
            int waveSizeUB) {
        this.orders = orders;
        this.aisles = aisles;
        this.nItems = nItems;
        this.waveSizeLB = waveSizeLB;
        this.waveSizeUB = waveSizeUB;
    }

    // ------------ Small helper graph (to replace NetworkX) -----------------
    static class Graph {
        Map<String, Set<String>> adj = new HashMap<>();
        Map<String, Integer> nodeWeight = new HashMap<>();
        int graphWeight = 0; // analogous to g.graph['weight'] in networkx

        void addNode(String n) {
            adj.putIfAbsent(n, new HashSet<>());
        }

        void setNodeWeight(String n, int w) {
            nodeWeight.put(n, w);
        }

        int getNodeWeight(String n) {
            return nodeWeight.getOrDefault(n, 0);
        }

        void addEdge(String u, String v) {
            addNode(u);
            addNode(v);
            adj.get(u).add(v);
            adj.get(v).add(u);
        }

        Set<String> nodes() {
            return adj.keySet();
        }

        Set<String> neighbors(String n) {
            return adj.getOrDefault(n, Collections.emptySet());
        }

        void removeNode(String n) {
            if (!adj.containsKey(n))
                return;
            for (String nb : new ArrayList<>(adj.get(n))) {
                adj.get(nb).remove(n);
            }
            adj.remove(n);
            nodeWeight.remove(n);
        }

        Set<String> isolates() {
            Set<String> res = new HashSet<>();
            for (var e : adj.entrySet()) {
                if (e.getValue().isEmpty())
                    res.add(e.getKey());
            }
            return res;
        }

        Graph subgraph(Set<String> keep) {
            Graph g = new Graph();
            for (String n : keep) {
                if (adj.containsKey(n)) {
                    g.addNode(n);
                    if (nodeWeight.containsKey(n))
                        g.setNodeWeight(n, nodeWeight.get(n));
                }
            }
            for (String u : keep) {
                for (String v : neighbors(u)) {
                    if (keep.contains(v))
                        g.addEdge(u, v);
                }
            }
            g.graphWeight = this.graphWeight;
            return g;
        }

        List<Set<String>> connectedComponents() {
            List<Set<String>> comps = new ArrayList<>();
            Set<String> vis = new HashSet<>();
            for (String s : nodes()) {
                if (vis.contains(s))
                    continue;
                Set<String> comp = new HashSet<>();
                Deque<String> dq = new ArrayDeque<>();
                dq.add(s);
                vis.add(s);
                while (!dq.isEmpty()) {
                    String u = dq.poll();
                    comp.add(u);
                    for (String v : neighbors(u)) {
                        if (!vis.contains(v)) {
                            vis.add(v);
                            dq.add(v);
                        }
                    }
                }
                comps.add(comp);
            }
            return comps;
        }
    }

    // --------------------------- Helpers -----------------------------------
    static String label(String prefix, String key) {
        return prefix + "[" + key + "]";
    }

    static String oLabel(int o) {
        return label("o", Integer.toString(o));
    }

    static String aLabel(int a) {
        return label("a", Integer.toString(a));
    }

    static int labelNumber(String name, int prefixSize) {
        // name like "o[123]": cut off first 2 chars "o[" and last char "]"
        return Integer.parseInt(name.substring(prefixSize, name.length() - 1));
    }

    // --------------------------- Instance ----------------------------------
    static class Instance {
        int O, I, A;
        int[][] u_oi; // O x I
        int[][] u_ai; // A x I

        List<Set<Integer>> order_items = new ArrayList<>(); // per order: items present
        List<Set<Integer>> aisle_items = new ArrayList<>(); // per aisle: items present

        List<Set<Integer>> item_orders; // per item: orders containing it
        List<Set<Integer>> item_aisles; // per item: aisles containing it

        Set<String> invalid_order_nodes = new HashSet<>();
        Set<String> trivial_nodes = new HashSet<>();

        int LB, UB;

        String input_file;
        Graph underlying_graph = new Graph();

        Instance(String inputFileOrNull) throws IOException {
            this.input_file = inputFileOrNull;
            BufferedReader br;
            if (inputFileOrNull == null) {
                br = new BufferedReader(new InputStreamReader(System.in));
            } else {
                br = new BufferedReader(new FileReader(inputFileOrNull));
            }

            // First line: O I A
            {
                String[] toks = nextNonEmptyLine(br).trim().split("\\s+");
                O = Integer.parseInt(toks[0]);
                I = Integer.parseInt(toks[1]);
                A = Integer.parseInt(toks[2]);
            }

            item_orders = new ArrayList<>(I);
            item_aisles = new ArrayList<>(I);
            for (int i = 0; i < I; i++) {
                item_orders.add(new HashSet<>());
                item_aisles.add(new HashSet<>());
            }

            // add nodes for bipartite sets
            for (int o = 0; o < O; o++)
                underlying_graph.addNode(oLabel(o));
            for (int a = 0; a < A; a++)
                underlying_graph.addNode(aLabel(a));

            // Read orders matrix: lines with sparse pairs (index, qty)
            u_oi = new int[O][I];
            for (int o = 0; o < O; o++) {
                String line = nextNonEmptyLine(br);
                int[] data = parseIntLine(line);
                // data in pattern: [k, i1, q1, i2, q2, ...] — but Python used data[1::2] as
                // indices (skip first?)
                // The original Python expects: count-like? We'll follow Python:
                // indices = data[1], data[3], ... ; qtys = data[2], data[4], ...
                // So we start from position 1.
                int sumQty = 0;
                Set<Integer> itemsHere = new HashSet<>();
                for (int p = 1; p + 1 < data.length; p += 2) {
                    int idx = data[p];
                    int qty = data[p + 1];
                    u_oi[o][idx] = qty;
                    if (qty > 0) {
                        item_orders.get(idx).add(o);
                        itemsHere.add(idx);
                        sumQty += qty;
                    }
                }
                order_items.add(itemsHere);
                underlying_graph.setNodeWeight(oLabel(o), sumQty);
            }

            // Read aisles matrix: also sparse pairs
            u_ai = new int[A][I];
            for (int a = 0; a < A; a++) {
                String line = nextNonEmptyLine(br);
                int[] data = parseIntLine(line);
                Set<Integer> itemsHere = new HashSet<>();
                for (int p = 1; p + 1 < data.length; p += 2) {
                    int idx = data[p];
                    int qty = data[p + 1];
                    u_ai[a][idx] = qty;
                    if (qty > 0) {
                        item_aisles.get(idx).add(a);
                        itemsHere.add(idx);
                    }
                }
                aisle_items.add(itemsHere);

                for (int o = 0; o < O; o++) {
                    // if aisle a shares any item with order o, add edge
                    if (!Collections.disjoint(aisle_items.get(a), order_items.get(o))) {
                        underlying_graph.addEdge(oLabel(o), aLabel(a));
                    }
                }
            }

            // Read LB, UB
            {
                String[] toks = nextNonEmptyLine(br).trim().split("\\s+");
                LB = Integer.parseInt(toks[0]);
                UB = Integer.parseInt(toks[1]);
            }

            br.close();
        }

        List<Graph> getComponents() {
            List<Graph> comps = new ArrayList<>();
            for (Set<String> compNodes : underlying_graph.connectedComponents()) {
                comps.add(underlying_graph.subgraph(compNodes));
            }
            return comps;
        }

        void clear_orders(int[] invalidOrders) {
            for (int o : invalidOrders)
                invalid_order_nodes.add(oLabel(o));
            for (String n : invalid_order_nodes)
                underlying_graph.removeNode(n);
            trivial_nodes = underlying_graph.isolates();
            for (String n : trivial_nodes)
                underlying_graph.removeNode(n);
            for (int o : invalidOrders)
                Arrays.fill(u_oi[o], 0);
        }

        double trivial_ub() {
            // Sort aisles by "number of items" (sum over items), desc. Compute
            // min(sum_items, UB)/(num_aisles)
            List<Integer> numAisleItems = new ArrayList<>();
            for (int a = 0; a < A; a++) {
                int tot = 0;
                for (int i = 0; i < I; i++)
                    tot += u_ai[a][i];
                numAisleItems.add(tot);
            }
            numAisleItems.sort(Comparator.reverseOrder());
            double best = 0.0;
            int sum = 0;
            for (int k = 0; k < numAisleItems.size(); k++) {
                sum += numAisleItems.get(k);
                double curr = Math.min(sum, UB) / (double) (k + 1);
                if (sum >= LB && curr > best) {
                    best = curr;
                    break;
                }
            }
            return best;
        }

        // ---- Parsing helpers
        private static String nextNonEmptyLine(BufferedReader br) throws IOException {
            String line;
            while ((line = br.readLine()) != null) {
                if (!line.trim().isEmpty())
                    return line;
            }
            throw new EOFException("Unexpected end of file.");
        }

        private static int[] parseIntLine(String line) {
            String[] toks = line.trim().split("\\s+");
            int[] arr = new int[toks.length];
            for (int i = 0; i < toks.length; i++)
                arr[i] = Integer.parseInt(toks[i]);
            return arr;
        }
    }

    // --------------------------- Solution ----------------------------------
    static class Solution {
        boolean empty = true;

        Map<String, Integer> variables = new HashMap<>();
        Set<String> order_nodes = new HashSet<>();
        List<Integer> orders = new ArrayList<>();
        Set<String> aisle_nodes = new HashSet<>();
        List<Integer> aisles = new ArrayList<>();

        int wave_orders = 0;
        int wave_items = 0;
        int wave_aisles = 0;
        double obj = Double.NEGATIVE_INFINITY;

        Graph underlying_graph;

        Solution(Map<String, Integer> vars, int numItems, Graph instanceGraph, boolean empty) {
            this.empty = empty;
            if (empty)
                return;
            this.variables = vars;
            this.wave_items = numItems;
            computeData();
            this.underlying_graph = instanceGraph.subgraph(union(order_nodes, aisle_nodes));
            this.underlying_graph.graphWeight = this.wave_items;
        }

        void computeData() {
            order_nodes.clear();
            orders.clear();
            aisle_nodes.clear();
            aisles.clear();

            for (var e : variables.entrySet()) {
                String key = e.getKey();
                int val = e.getValue();
                if (val == 0)
                    continue;
                char c = key.charAt(0);
                if (c == 'o') {
                    order_nodes.add(key);
                    orders.add(labelNumber(key, 2));
                } else if (c == 'a') {
                    aisle_nodes.add(key);
                    aisles.add(labelNumber(key, 2));
                }
            }
            wave_orders = orders.size();
            wave_aisles = aisles.size();
            obj = wave_aisles > 0 ? (wave_items * 1.0 / wave_aisles) : (wave_items);
        }

        List<Graph> getComponents() {
            List<Graph> comps = new ArrayList<>();
            for (Set<String> compNodes : underlying_graph.connectedComponents()) {
                comps.add(underlying_graph.subgraph(compNodes));
            }
            return comps;
        }

        void restrict_solution(Graph comp) {
            Set<String> compNodes = comp.nodes();
            for (String node : new HashSet<>(order_nodes)) {
                if (!compNodes.contains(node))
                    variables.put(node, 0);
            }
            for (String node : new HashSet<>(aisle_nodes)) {
                if (!compNodes.contains(node))
                    variables.put(node, 0);
            }
            this.wave_items = comp.graphWeight;
            computeData();
            this.underlying_graph = comp;
        }

        private static Set<String> union(Set<String> a, Set<String> b) {
            Set<String> r = new HashSet<>(a);
            r.addAll(b);
            return r;
        }
    }

    static void bestSubSolution(Solution sol, int LB) {
        List<Graph> comps = sol.getComponents();
        if (comps.size() <= 1)
            return;

        double bestLocal = Double.NEGATIVE_INFINITY;
        Graph bestComp = null;
        int bestCompSize = 0;

        for (Graph comp : comps) {
            Set<String> compNodes = comp.nodes();
            Set<String> compOrders = compNodes.stream().filter(n -> n.startsWith("o[")).collect(Collectors.toSet());
            Set<String> compAisles = compNodes.stream().filter(n -> n.startsWith("a[")).collect(Collectors.toSet());

            int compWaveItems = 0;
            for (String on : compOrders)
                compWaveItems += comp.getNodeWeight(on);
            int compWaveAisles = compAisles.size();

            if (compWaveAisles > 0 && compWaveItems >= LB) {
                double compObj = compWaveItems / (double) compWaveAisles;
                if (compObj > bestLocal) {
                    bestLocal = compObj;
                    bestComp = comp;
                    bestCompSize = compWaveItems;
                }
            }
        }

        if (bestComp != null && bestLocal > sol.obj) {
            bestComp.graphWeight = bestCompSize;
            System.out.printf("- Found a better sub-solution with obj = %.3f;%n", bestLocal);
            sol.restrict_solution(bestComp);
        }
    }

    // -------------------------- CPLEX Solution ------------------------------
    static class CPLEXSolution extends Solution {
        CPLEXSolution(IloCplex cplex,
                Map<String, IloNumVar> nameToVar,
                IloNumExpr waveItemsExpr,
                Graph instanceGraph,
                boolean empty) throws IloException {
            super(
                    empty ? null : extractSolutionValues(cplex, nameToVar),
                    empty ? 0 : (int) Math.round(cplex.getValue(waveItemsExpr)),
                    instanceGraph,
                    empty);
        }

        private static Map<String, Integer> extractSolutionValues(IloCplex cplex,
                Map<String, IloNumVar> nameToVar)
                throws IloException {
            Map<String, Integer> vals = new HashMap<>();
            int cnt = 0;
            for (var e : nameToVar.entrySet()) {
                double v = cplex.getValue(e.getValue());
                vals.put(e.getKey(), (int) Math.round(v));
            }
            return vals;
        }
    }

    // -------------------- Preprocessing: min aisles cover per order ----------
    static int[] minOrdersCover(Instance inst) throws IloException {
        int[] result = new int[inst.O];
        IloCplex setModel = new IloCplex();
        setModel.setParam(IloCplex.Param.Threads, 16);
        setModel.setOut(null);
        setModel.setWarning(null);

        // Decision: select aisles
        IloNumVar[] Avars = setModel.boolVarArray(inst.A);
        for (int a = 0; a < inst.A; a++)
            Avars[a].setName(aLabel(a));

        // We'll rebuild constraints per order
        List<IloRange> currentCons = new ArrayList<>();

        for (int o = 0; o < inst.O; o++) {
            // Add covering constraints for each item in order o
            for (int i : inst.order_items.get(o)) {
                IloLinearNumExpr lhs = setModel.linearNumExpr();
                for (int a : inst.item_aisles.get(i)) {
                    int cap = inst.u_ai[a][i];
                    if (cap > 0)
                        lhs.addTerm(cap, Avars[a]);
                }
                int demand = inst.u_oi[o][i];
                IloRange c = setModel.addGe(lhs, demand);
                currentCons.add(c);
            }
            // Objective: minimize sum A
            IloLinearNumExpr obj = setModel.linearNumExpr();
            for (IloNumVar v : Avars)
                obj.addTerm(1.0, v);
            setModel.addMinimize(obj);

            boolean solved = setModel.solve();
            if (solved && setModel.getMIPRelativeGap() == 0.0) {
                result[o] = (int) Math.round(setModel.getObjValue());
            } else {
                result[o] = 0; // infeasible or non-optimal -> mark as invalid later if 0
            }

            // Clear constraints & objective for next order
            setModel.remove(setModel.getObjective());
            if (!currentCons.isEmpty()) {
                setModel.remove(currentCons.toArray(new IloRange[0]));
                currentCons.clear();
            }
        }

        setModel.end();
        return result;
    }

    // ------------------------------ Main ------------------------------------
    public static void main(String[] args) {
        long startTime = System.currentTimeMillis();
        String instancePath = (args.length >= 1) ? args[0] : null;

        try {
            Instance inst = new Instance(instancePath);

            // ---- Preprocess
            int[] minCovers = minOrdersCover(inst);
            // mark invalid orders (min cover == 0)
            List<Integer> invalid = new ArrayList<>();
            for (int o = 0; o < inst.O; o++)
                if (minCovers[o] == 0)
                    invalid.add(o);
            inst.clear_orders(invalid.stream().mapToInt(i -> i).toArray());

            System.out.printf("Preprocessing completed after %.2fs%n",
                    (System.currentTimeMillis() - startTime) / 1000.0);

            // ---- Build Dinkelbach model
            IloCplex cplex = new IloCplex();
            cplex.setOut(null); // silence; set to System.out to debug
            cplex.setParam(IloCplex.Param.Threads, 16);
            cplex.setParam(IloCplex.Param.Emphasis.MIP, IloCplex.MIPEmphasis.Feasibility); // CPX_MIPEMPHASIS_FEASIBILITY

            // Variables with names & maps for quick access by name
            Map<String, IloNumVar> nameToVar = new HashMap<>();

            // Orders
            IloNumVar[] Ovars = cplex.boolVarArray(inst.O);
            for (int o = 0; o < inst.O; o++) {
                String nm = oLabel(o);
                Ovars[o].setName(nm);
                nameToVar.put(nm, Ovars[o]);
            }
            // Aisles
            IloNumVar[] Avars = cplex.boolVarArray(inst.A);
            for (int a = 0; a < inst.A; a++) {
                String nm = aLabel(a);
                Avars[a].setName(nm);
                nameToVar.put(nm, Avars[a]);
            }

            // wave_items expression
            IloLinearNumExpr waveItemsExpr = cplex.linearNumExpr();
            for (int o = 0; o < inst.O; o++) {
                int sum = 0;
                for (int i = 0; i < inst.I; i++)
                    sum += inst.u_oi[o][i];
                if (sum != 0)
                    waveItemsExpr.addTerm(sum, Ovars[o]);
            }

            // Operational limits on items
            cplex.addGe(waveItemsExpr, inst.LB);
            cplex.addLe(waveItemsExpr, inst.UB);

            // Capacity constraints per item
            for (int i = 0; i < inst.I; i++) {
                IloLinearNumExpr dem = cplex.linearNumExpr();
                for (int o : inst.item_orders.get(i)) {
                    int demand = inst.u_oi[o][i];
                    if (demand > 0)
                        dem.addTerm(demand, Ovars[o]);
                }
                IloLinearNumExpr cap = cplex.linearNumExpr();
                for (int a : inst.item_aisles.get(i)) {
                    int capacity = inst.u_ai[a][i];
                    if (capacity > 0)
                        cap.addTerm(capacity, Avars[a]);
                }
                cplex.addLe(dem, cap);
            }

            // Remove trivial/invalid nodes:
            for (String node : inst.trivial_nodes) {
                nameToVar.remove(node);
            }
            for (String node : inst.invalid_order_nodes) {
                nameToVar.remove(node);
            }

            // Utility to fix/reset variables
            Runnable resetRemoved = () -> {
            }; // will be set later
            Set<String> removed = new HashSet<>();

            // --------------- Dinkelbach loop ---------------
            CPLEXSolution bestSol = new CPLEXSolution(null, null, null, null, true);
            Graph remainingGraph = inst.underlying_graph; // search space

            final double TOL = 1e-6;
            double OPT_LB = Math.max(inst.LB / (double) Math.max(inst.A, 1), TOL);
            double OPT_UB = inst.trivial_ub();
            double lambda = OPT_LB;

            double elapsedTime = (System.currentTimeMillis() - startTime) / 1000.0;
            double totalTime = 60 * 10 - elapsedTime; // 10 minutes minus preprocessing
            double timeTolerance = 20.0;
            double timeLimit = totalTime;

            cplex.setParam(IloCplex.DoubleParam.TimeLimit, Math.min(30.0, totalTime));
            // Upper cutoff -> use as an incumbent cutoff surrogate:
            // cplex.setParam(IloCplex.DoubleParam.MIP.Limits.LowerObjStop, 0.0);
            cplex.setParam(IloCplex.Param.MIP.Limits.Solutions, 10);

            double dinkObj = Double.POSITIVE_INFINITY;

            System.out.println("\nStarting Dinkelbach search...");
            long dinkStart = System.currentTimeMillis();
            long iterStart = System.currentTimeMillis();

            while (Math.abs(dinkObj) > TOL) {
                double GAP = (OPT_UB - OPT_LB) / Math.max(OPT_LB, 1e-12);
                System.out.printf("Current interval = [%.3f:%.3f]; gap = %.3f%%; λ = %.2f; ",
                        OPT_LB, OPT_UB, 100.0 * GAP, lambda);

                // Maximize wave_items - lambda * wave_aisles
                IloLinearNumExpr obj = cplex.linearNumExpr();
                obj.add(waveItemsExpr);

                // wave_aisles expression
                IloLinearNumExpr waveAislesExprObj = cplex.linearNumExpr();
                for (int a = 0; a < inst.A; a++)
                    waveAislesExprObj.addTerm(-lambda, Avars[a]);
                obj.add(waveAislesExprObj);
                cplex.addMaximize(obj);

                boolean solved = cplex.solve();
                if (!solved)
                    throw new RuntimeException("No solution found!");
                // Update time limits
                double iterDur = (System.currentTimeMillis() - iterStart) / 1000.0;
                iterStart = System.currentTimeMillis();
                timeLimit = Math.max(0, timeLimit - iterDur);
                cplex.setParam(IloCplex.DoubleParam.TimeLimit, Math.max(0.0, timeLimit - timeTolerance));

                dinkObj = Math.min(dinkObj, cplex.getObjValue());
                double elapsed = (System.currentTimeMillis() - dinkStart) / 1000.0;
                System.out.printf("dinkelbach obj = %.6f; elapsed %.2fs of %.2fs;%n",
                        dinkObj, elapsed, Math.max(0.0, totalTime - timeTolerance));

                // current objective = wave_items / wave_aisles (values at incumbent)
                double waveItemsVal = cplex.getValue(waveItemsExpr);
                double waveAislesVal = cplex.getValue(waveAislesExprObj);
                double currentObj = (waveAislesVal > 0.0) ? (waveItemsVal / waveAislesVal) : waveItemsVal;

                boolean improved = currentObj > bestSol.obj;
                if (improved) {
                    // cplex.setParam(IloCplex.DoubleParam.MIP.Limits.LowerObjStop, 0.0);
                    cplex.setParam(IloCplex.Param.MIP.Limits.Solutions, 10);
                    bestSol = new CPLEXSolution(cplex, nameToVar, waveItemsExpr, remainingGraph, false);
                    System.out.printf("- Found a new best solution with obj = %.3f;%n", bestSol.obj);
                    bestSubSolution(bestSol, inst.LB);
                }

                // remove objective to re-set next iteration
                cplex.delete(cplex.getObjective());

                if (timeLimit <= timeTolerance)
                    break;

                // Update search interval
                OPT_LB = Math.max(OPT_LB, bestSol.obj);
                lambda = OPT_LB;

                // Local optimum / pruning opportunities
                if (!improved) {
                    System.out.println("- Reached a local optimum!");
                    if (!removed.isEmpty()) {
                        // restore
                        for (String nm : removed) {
                            IloNumVar v = nameToVar.get(nm);
                            if (v != null)
                                v.setUB(1.0);
                        }
                        System.out.printf("- %d fixed variables were restored;%n", removed.size());
                        removed.clear();
                        remainingGraph = inst.underlying_graph;
                        dinkObj = Double.POSITIVE_INFINITY;
                        continue;
                    }

                    // Enforce some improvement
                    // wave_items - lambda * wave_aisles >= TOL
                    IloLinearNumExpr improve = cplex.linearNumExpr();
                    improve.add(waveItemsExpr);

                    IloLinearNumExpr waveAislesExpr = cplex.linearNumExpr();
                    for (int a = 0; a < inst.A; a++)
                        waveAislesExpr.addTerm(-lambda, Avars[a]);

                    improve.add(waveAislesExpr);
                    cplex.addGe(improve, TOL);

                    // cplex.setParam(IloCplex.DoubleParam.MIP.Limits.LowerObjStop, inst.UB);
                    cplex.setParam(IloCplex.Param.MIP.Limits.Solutions, 1000000);

                    if (dinkObj < -TOL) {
                        System.out.println("- The solver stopped too early, restarting the iteration;");
                        dinkObj = Double.POSITIVE_INFINITY;
                    }
                }
            }

            System.out.printf("...Dinkelbach search stopped after %.2fs.%n",
                    (System.currentTimeMillis() - dinkStart) / 1000.0);

            if (bestSol.empty)
                throw new RuntimeException("No feasible solution found!");

            if (inst.input_file != null) {
                System.out.printf("%nBest solution found for instance %s:%n", inst.input_file);
            } else {
                System.out.printf("%nBest solution found for instance %dx%dx%d:%n", inst.O, inst.I, inst.A);
            }
            System.out.printf(" - LB = %d, UB = %d;%n", inst.LB, inst.UB);
            System.out.printf(" - %d/%d orders", bestSol.orders.size(), inst.O);
            System.out.printf(" - %d/%d aisles", bestSol.aisles.size(), inst.A);
            System.out.printf(" - %d items;%n", bestSol.wave_items);
            System.out.printf(" - total time: %.2fs;%n", (System.currentTimeMillis() - startTime) / 1000.0);
            System.out.printf(" - obj: %.2f;%n", bestSol.obj);

            cplex.end();

        } catch (IloException e) {
            System.out.println("Error:\n" + e.getMessage());
        } catch (Exception e) {
            System.out.println("Encountered an error: " + e.getMessage());
        }
    }

    public ChallengeSolution solve(StopWatch stopWatch) {
        // Implement your solution here
        return null;
    }

    /*
     * Get the remaining time in seconds
     */
    protected long getRemainingTime(StopWatch stopWatch) {
        return Math.max(
                TimeUnit.SECONDS.convert(MAX_RUNTIME - stopWatch.getTime(TimeUnit.MILLISECONDS), TimeUnit.MILLISECONDS),
                0);
    }

    protected boolean isSolutionFeasible(ChallengeSolution challengeSolution) {
        Set<Integer> selectedOrders = challengeSolution.orders();
        Set<Integer> visitedAisles = challengeSolution.aisles();
        if (selectedOrders == null || visitedAisles == null || selectedOrders.isEmpty() || visitedAisles.isEmpty()) {
            return false;
        }

        int[] totalUnitsPicked = new int[nItems];
        int[] totalUnitsAvailable = new int[nItems];

        // Calculate total units picked
        for (int order : selectedOrders) {
            for (Map.Entry<Integer, Integer> entry : orders.get(order).entrySet()) {
                totalUnitsPicked[entry.getKey()] += entry.getValue();
            }
        }

        // Calculate total units available
        for (int aisle : visitedAisles) {
            for (Map.Entry<Integer, Integer> entry : aisles.get(aisle).entrySet()) {
                totalUnitsAvailable[entry.getKey()] += entry.getValue();
            }
        }

        // Check if the total units picked are within bounds
        int totalUnits = Arrays.stream(totalUnitsPicked).sum();
        if (totalUnits < waveSizeLB || totalUnits > waveSizeUB) {
            return false;
        }

        // Check if the units picked do not exceed the units available
        for (int i = 0; i < nItems; i++) {
            if (totalUnitsPicked[i] > totalUnitsAvailable[i]) {
                return false;
            }
        }

        return true;
    }

    protected double computeObjectiveFunction(ChallengeSolution challengeSolution) {
        Set<Integer> selectedOrders = challengeSolution.orders();
        Set<Integer> visitedAisles = challengeSolution.aisles();
        if (selectedOrders == null || visitedAisles == null || selectedOrders.isEmpty() || visitedAisles.isEmpty()) {
            return 0.0;
        }
        int totalUnitsPicked = 0;

        // Calculate total units picked
        for (int order : selectedOrders) {
            totalUnitsPicked += orders.get(order).values().stream()
                    .mapToInt(Integer::intValue)
                    .sum();
        }

        // Calculate the number of visited aisles
        int numVisitedAisles = visitedAisles.size();

        // Objective function: total units picked / number of visited aisles
        return (double) totalUnitsPicked / numVisitedAisles;
    }
}

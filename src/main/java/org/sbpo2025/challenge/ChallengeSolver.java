package org.sbpo2025.challenge;

import org.apache.commons.lang3.time.StopWatch;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

import ilog.concert.*;
import ilog.cplex.*;

public class ChallengeSolver {

    private record Model(IloCplex cplex, IloNumVar[] Ovars, IloNumVar[] Avars, Map<String, IloNumVar> nameToVar,
                         IloLinearNumExpr waveItemsExpr, IloLinearNumExpr waveAislesExpr, Set<String> removed) {
    }

    private static Model model;
    private final long MAX_RUNTIME = 600000; // milliseconds; 10 minutes
    private final double TOL = 1e-6;
    private Instance inst;

    public ChallengeSolver(Instance instance) {
        this.inst = instance;
    }

    public static boolean bestSubSolution(ChallengeSolution sol, int LB) {
        List<Graph> comps = sol.getComponents();
        if (comps.size() <= 1)
            return false;

        double bestLocal = Double.NEGATIVE_INFINITY;
        Graph bestComp = null;
        int bestCompSize = 0;

        for (Graph comp : comps) {
            Set<String> compNodes = comp.getNodes();
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
            System.out.printf("\t- Found a better sub-solution with obj = %.3f;\n", bestLocal);
            sol.restrict_solution(bestComp);
            return true;
        }
        return false;
    }

    // -------------------- Preprocessing: min aisles cover per order ----------
    public static int[] minOrdersCover(Instance inst) throws IloException {
        int[] result = new int[inst.O];
        try (IloCplex setModel = new IloCplex()) {
            setModel.setParam(IloCplex.Param.Threads, 6);
            setModel.setOut(null);
            setModel.setWarning(null);

            // Decision: select aisles
            IloNumVar[] Avars = setModel.boolVarArray(inst.A);
            for (int a = 0; a < inst.A; a++)
                Avars[a].setName(Helpers.aLabel(a));

            // We'll rebuild constraints per order
            List<IloRange> currentCons = new ArrayList<>();

            for (int o = 0; o < inst.O; o++) {
                // Add covering constraints for each item in order o
                for (int i : inst.order_items.get(o)) {
                    IloLinearNumExpr lhs = setModel.linearNumExpr();
                    for (int a : inst.item_aisles.get(i)) {
                        int cap = inst.u_ai.get(a).getOrDefault(i, 0);
                        if (cap > 0)
                            lhs.addTerm(cap, Avars[a]);
                    }
                    int demand = inst.u_oi.get(o).getOrDefault(i, 0);
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

                // Clear constraints and objective for the next order
                setModel.remove(setModel.getObjective());
                if (!currentCons.isEmpty()) {
                    setModel.remove(currentCons.toArray(new IloRange[0]));
                    currentCons.clear();
                }
            }

            setModel.end();
        }
        return result;
    }

    // -------------------- Preprocessing: LP-relaxation ----------------------
    private int getLPSolution(Set<String> vars) {
        try {
            IloCplex cplex = new IloCplex();
            cplex.setOut(new PrintStream(new FileOutputStream("cplex_output_lp.txt")));
            cplex.setParam(IloCplex.Param.Threads, 6);

            Map<String, IloNumVar> nameToVar = new HashMap<>();

            IloNumVar[] Ovars = cplex.numVarArray(inst.O, 0.0, 1.0);
            for (int o = 0; o < inst.O; o++) {
                String nm = Helpers.oLabel(o);
                Ovars[o].setName(nm);
                nameToVar.put(nm, Ovars[o]);
            }
            // Aisles
            IloNumVar[] Avars = cplex.numVarArray(inst.A, 0.0, 1.0);
            for (int a = 0; a < inst.A; a++) {
                String nm = Helpers.aLabel(a);
                Avars[a].setName(nm);
                nameToVar.put(nm, Avars[a]);
            }

            // wave_items expression
            IloLinearNumExpr waveItemsExpr = cplex.linearNumExpr();
            for (int o = 0; o < inst.O; o++) {
                int sum = 0;
                for (int val : inst.u_oi.get(o).values())
                    sum += val;
                if (sum != 0)
                    waveItemsExpr.addTerm(sum, Ovars[o]);
            }

            // Operational limits on items
            cplex.addGe(waveItemsExpr, inst.UB);
            cplex.addLe(waveItemsExpr, inst.UB);

            // Capacity constraints per item
            for (int i = 0; i < inst.I; i++) {
                IloLinearNumExpr dem = cplex.linearNumExpr();
                for (int o : inst.item_orders.get(i)) {
                    int demand = inst.u_oi.get(o).getOrDefault(i, 0);
                    if (demand > 0)
                        dem.addTerm(demand, Ovars[o]);
                }
                IloLinearNumExpr cap = cplex.linearNumExpr();
                for (int a : inst.item_aisles.get(i)) {
                    int capacity = inst.u_ai.get(a).get(i);
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

            cplex.setParam(IloCplex.Param.MIP.Pool.Capacity, 3);
            IloLinearNumExpr obj = cplex.linearNumExpr();
            // wave_aisles expression
            IloLinearNumExpr waveAislesExprObj = cplex.linearNumExpr();
            for (int a = 0; a < inst.A; a++)
                waveAislesExprObj.addTerm(1.0, Avars[a]);
            obj.add(waveAislesExprObj);
            cplex.addMinimize(obj);

            boolean solved = cplex.solve();
            if (!solved)
                throw new RuntimeException("No solution found!");
            for (String key : nameToVar.keySet()) {
                if (cplex.getValue(nameToVar.get(key)) > 0.1) {
                    vars.add(key);
                }
            }

            return nameToVar.size() - vars.size();
        } catch (IloException e) {
            System.out.println("Error:\n" + e.getMessage());
        } catch (Exception e) {
            System.out.println("Encountered an error: " + e.getMessage());
        }
        assert (false);
        return -1;
    }

    private void coveringHeuristic() throws IloException {
        System.out.println("Computing invalid orders and isolated vertices...");
        int[] minCovers = minOrdersCover(inst);
        // Mark invalid orders (min cover == 0):
        var invalid = new ArrayList<Integer>();
        for (int o = 0; o < inst.O; o++)
            if (minCovers[o] == 0)
                invalid.add(o);
        int isolated = inst.clear_orders(invalid);
        System.out.printf("\t- %d orders and %d aisles were invalidated;\n", inst.invalid_order_nodes.size(), isolated);
    }

    private void preprocess(Set<String> varInLP) throws IloException {
        coveringHeuristic();

        System.out.println("Running Linear Relaxation...");
        int removedByLP = getLPSolution(varInLP);
        System.out.println("\t- LP-relaxation removed " + removedByLP + " variables;");
    }

    private static Set<String> setUB(Set<String> varNames, double UB, int limit) throws IloException {
        Set<String> removed = new HashSet<>();
        int count = 0;
        for (String var : varNames) {
            IloNumVar v = model.nameToVar().get(var);
            if (v != null)
                v.setUB(UB);
            removed.add(var);
            count++;
            if (count == limit)
                break;
        }
        return removed;
    }

    private static Set<String> setUB(Set<String> varNames, double UB) throws IloException {
        return setUB(varNames, UB, varNames.size());
    }

    private void buildModel(HashSet<String> varInLP) throws IloException, FileNotFoundException {
        System.out.println("\nBuilding Dinkelbach model...");
        IloCplex cplex = new IloCplex();
        cplex.setOut(new PrintStream(new FileOutputStream("cplex_output.txt")));
        cplex.setParam(IloCplex.Param.Threads, 6);
        cplex.setParam(IloCplex.Param.Emphasis.MIP, IloCplex.MIPEmphasis.HiddenFeas);
        Map<String, IloNumVar> nameToVar = new HashMap<>(); // variables with names and maps for quick access by name

        // Orders
        IloNumVar[] Ovars = cplex.boolVarArray(inst.O);
        for (int o = 0; o < inst.O; o++) {
            String nm = Helpers.oLabel(o);
            Ovars[o].setName(nm);
            nameToVar.put(nm, Ovars[o]);
        }
        // Aisles
        IloNumVar[] Avars = cplex.boolVarArray(inst.A);
        IloLinearNumExpr waveAislesExpr = cplex.linearNumExpr();
        for (int a = 0; a < inst.A; a++) {
            String nm = Helpers.aLabel(a);
            Avars[a].setName(nm);
            nameToVar.put(nm, Avars[a]);
            waveAislesExpr.addTerm(1.0, Avars[a]);
        }

        // wave_items expression
        IloLinearNumExpr waveItemsExpr = cplex.linearNumExpr();
        for (int o = 0; o < inst.O; o++) {
            int sum = 0;
            for (int val : inst.u_oi.get(o).values())
                sum += val;
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
                int demand = inst.u_oi.get(o).getOrDefault(i, 0);
                if (demand > 0)
                    dem.addTerm(demand, Ovars[o]);
            }
            IloLinearNumExpr cap = cplex.linearNumExpr();
            for (int a : inst.item_aisles.get(i)) {
                int capacity = inst.u_ai.get(a).get(i);
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

        // Fix variables removed by the LP relaxation (if any):
        var removed = new HashSet<>(nameToVar.keySet());
        removed.removeAll(varInLP);

        // Create the model record:
        model = new Model(cplex, Ovars, Avars, nameToVar, waveItemsExpr, waveAislesExpr, removed);
        setUB(removed, 0.0);
    }

    public ChallengeSolution solve(StopWatch stopWatch) {
        try {
            // ---- Preprocess
            inst.objLB = Math.max(inst.LB / Math.max(inst.A, 1.0), TOL);
            long start = getCurrentTime(stopWatch);
            var varInLP = new HashSet<String>();
            preprocess(varInLP);
            System.out.printf("- Preprocessing completed after %.2fs.\n", (getCurrentTime(stopWatch) - start) / 1E3);

            // ---- Build Dinkelbach model
            start = getCurrentTime(stopWatch);
            buildModel(varInLP);
            System.out.printf("- Model build completed after %.2fs.\n", (getCurrentTime(stopWatch) - start) / 1E3);

            // --------------- Dinkelbach loop ---------------
            CPLEXSolution bestSol = new CPLEXSolution();
            Graph remainingGraph = inst.underlying_graph; // search space

            double OPT_LB = inst.objLB;
            double OPT_UB = inst.trivial_ub();
            double lambda = OPT_LB;

            double elapsedTime = getCurrentTime(stopWatch) / 1E3;
            double totalTime = 60 * 10 - elapsedTime; // 10 minutes minus preprocessing
            double timeTolerance = 5.0;
            double timeLimit = totalTime;

            model.cplex().setParam(IloCplex.DoubleParam.TimeLimit, Math.min(30.0, totalTime));
            model.cplex().setParam(IloCplex.Param.MIP.Pool.Capacity, 2);

            System.out.println("\nStarting Dinkelbach search...");
            long dinkStart = System.currentTimeMillis();
            long iterStart = System.currentTimeMillis();
            boolean is_first_iteration = true;

            var finalIter = true;
            var superRemoved = new HashSet<String>();
            var secondProcess = true;
            while (true) {
                if (!bestSol.empty) {
                    var aux = bestSol.wave_items;
                    var aux2 = bestSol.obj;
                    for (int a = 0; a < inst.A; a++) {
                        var adj = inst.u_ai.get(a);
                        if (adj.isEmpty()) continue;
                        var cap = adj.values().stream().mapToInt(Integer::intValue).sum();
                        if (aux - cap >= inst.LB && cap < aux2) {
                            superRemoved.add(Helpers.aLabel(a));
                            inst.u_ai.get(a).clear();
                        }
                    }
                    setUB(superRemoved, 0.0); // reaally remove some variables

                    if (secondProcess) {
                        coveringHeuristic();
                        for (var invalid : inst.invalid_order_nodes)
                            model.cplex().remove(model.nameToVar().get(invalid));
                        for (var trivial : inst.trivial_nodes)
                            model.cplex().remove(model.nameToVar().get(trivial));
                        secondProcess = false;
                    }
                }
                System.out.println("\t- Variables were super removed = " + superRemoved.size() + " out of " + model.nameToVar().size() + " aisles: " + inst.A);
                double GAP = 100 * (OPT_UB - OPT_LB) / Math.max(OPT_LB, TOL);
                System.out.printf("Current interval = [%.3f:%.3f]; gap = %.3f%%; λ = %.2f; ", OPT_LB, OPT_UB, GAP,
                        lambda);

                // Create the objective function (wave_items - lambda * wave_aisles) object:
                IloLinearNumExpr obj = model.cplex().linearNumExpr();
                if (!is_first_iteration) {
                    obj.add(model.waveItemsExpr());
                }
                IloLinearNumExpr waveAislesExprObj = model.cplex().linearNumExpr();
                for (var var : model.Avars())
                    waveAislesExprObj.addTerm(-lambda, var);
                obj.add(waveAislesExprObj);
                model.cplex().addMaximize(obj);

                // Solve the current model:
                boolean solved = model.cplex().solve();

                // Validate the LP relaxation heuristic:
                if (!solved && is_first_iteration) {
                    is_first_iteration = false;
                    setUB(model.removed(), 1.0);
                    model.removed().clear();
                    model.cplex.remove(model.cplex.getObjective());
                    System.out
                            .println("\n\t- Variables selected by the LP relaxation do not yield a feasible solution;");
                    continue;
                }

                if (!solved)
                    throw new RuntimeException("No Dinkelbach solution found!");

                // Update the time limits:
                double iterDur = (System.currentTimeMillis() - iterStart) / 1E3;
                iterStart = System.currentTimeMillis();
                timeLimit = Math.max(0, timeLimit - iterDur);
                model.cplex().setParam(IloCplex.DoubleParam.TimeLimit, Math.max(0.0, timeLimit - timeTolerance));

                // Get the Dinkelbach objective value:
                double dinkObj = model.cplex().getObjValue();
                double elapsed = (System.currentTimeMillis() - dinkStart) / 1E3;
                double total = Math.max(0.0, totalTime - timeTolerance);
                System.out.printf("dinkelbach obj = %.6f; elapsed %.2fs of %.2fs;\n", dinkObj, elapsed, total);

                // Compute the current incumbent's objective (wave_items / wave_aisles):
                int waveItemsVal = (int) Math.round(model.cplex().getValue(model.waveItemsExpr()));
                int waveAislesVal = (int) Math.round(model.cplex().getValue(model.waveAislesExpr()));
                double currentObj = (double) waveItemsVal / Math.max(waveAislesVal, 1);
                if (!is_first_iteration && model.removed().isEmpty() && currentObj < -TOL) {
                    throw new IllegalStateException("Negative Objective Function!");
                }

                boolean improved = currentObj > bestSol.obj + 1.0 / inst.UB - TOL;
                if (improved) {
                    model.cplex().setParam(IloCplex.DoubleParam.MIP.Limits.LowerObjStop, 1.0 / inst.UB);
                    model.cplex().setParam(IloCplex.Param.MIP.Limits.Solutions, 5);
                    bestSol = new CPLEXSolution(model.cplex(), model.nameToVar(), waveItemsVal, remainingGraph);
                    if (!is_first_iteration
                            && Math.abs(bestSol.wave_items - bestSol.wave_aisles * lambda - dinkObj) >= TOL) {
                        throw new IllegalStateException(
                                "Invariant violated: wave_items - wave_aisles * lambda != dinkObj");
                    }
                    System.out.printf("\t- Found a new best solution with obj = %.3f;\n", bestSol.obj);
                    localSearchImprove(bestSol);
                    if (bestSubSolution(bestSol, inst.LB)) {
                        ArrayList<IloNumVar> vars = new ArrayList<>();
                        for (Integer aisle : bestSol.get_aisles()) {
                            vars.add(model.nameToVar().get(Helpers.aLabel(aisle)));
                        }
                        for (Integer order : bestSol.get_orders()) {
                            vars.add(model.nameToVar().get(Helpers.oLabel(order)));
                        }
                        double[] values = new double[vars.size()];
                        Arrays.fill(values, 1.0);
                        var varArray = vars.toArray(new IloNumVar[0]);
                        model.cplex().addMIPStart(varArray, values, IloCplex.MIPStartEffort.Repair, "MyStart");
                    }
                }

                // First iteration reset:
                if (is_first_iteration) {
                    is_first_iteration = false;
                    setUB(model.removed(), 1.0);
                    model.removed().clear();
                }

                // Remove objective to re-set the next iteration:
                model.cplex().delete(model.cplex().getObjective());

                if (timeLimit <= timeTolerance)
                    break;

                // Update search interval
                OPT_LB = Math.max(OPT_LB, bestSol.obj);
                lambda = OPT_LB;

                // Local optimum / pruning opportunities
                if (!improved) {
                    System.out.println("\t- Reached a local optimum!");
                    if (!model.removed().isEmpty()) {
                        // restore
                        Set<String> solution_nodes = new HashSet<>();
                        solution_nodes.addAll(bestSol.aisle_nodes);
                        solution_nodes.addAll(bestSol.order_nodes);
                        var mapDistance = inst.underlying_graph.compute_distance_from_set(solution_nodes);
                        int resetTotal = 0;
                        int max = Math.max(5000, Math.min(11000, model.nameToVar().size() / 5));
                        if (model.removed().size() - max < model.nameToVar().size() / 5)
                            max = model.removed().size();
                        do {
                            var minDistance = model.removed().stream().mapToInt(
                                            var -> mapDistance.getOrDefault(var, Integer.MAX_VALUE)).min()
                                    .orElse(Integer.MAX_VALUE);
                            Set<String> restore = model.removed().stream().filter(
                                            var -> mapDistance.getOrDefault(var, Integer.MAX_VALUE) <= minDistance)
                                    .collect(Collectors.toCollection(HashSet::new));
                            restore = setUB(restore, 1.0, max - resetTotal);
                            model.removed().removeAll(restore);

                            System.out.printf("\t- %d fixed variables from distance <= %d were restored;\n",
                                    restore.size(),
                                    minDistance);
                            System.out.println("\t- " + model.removed().size() + " variables out of "
                                    + model.nameToVar().size() + " are fixed;");
                            resetTotal += restore.size();
                        } while ((resetTotal < max || model.removed().size() <= 1000) && !model.removed().isEmpty());

                        var toKeep = inst.underlying_graph.getNodes();
                        model.removed().forEach(toKeep::remove);
                        remainingGraph = inst.underlying_graph.subgraph(toKeep);
                    } else {
                        if (finalIter) {
                            finalIter = false;
                            System.out.println("Running final iteration (resetting all parameters)...");
                            model.cplex().setParam(IloCplex.Param.MIP.Pool.Capacity, 2100000000);
                            model.cplex().setParam(IloCplex.DoubleParam.MIP.Limits.LowerObjStop, -1E+75);
                            model.cplex().setParam(IloCplex.Param.MIP.Limits.Solutions, 9223372036800000000L);
                            model.cplex().setParam(IloCplex.Param.Emphasis.MIP, IloCplex.MIPEmphasis.Optimality);
                            continue;
                        }
                        if (model.cplex().getStatus() == IloCplex.Status.Optimal)
                            break;
                        throw new IllegalStateException("Solution should be optimal!");
                    }
                } else if (model.nameToVar().size() > 1997) {
                    Set<String> solution_nodes = new HashSet<>();
                    solution_nodes.addAll(bestSol.aisle_nodes);
                    solution_nodes.addAll(bestSol.order_nodes);
                    Map<String, Integer> mapDistance = remainingGraph.compute_distance_from_set(solution_nodes);
                    Set<String> to_remove = new HashSet<>();
                    Set<String> to_keep = new HashSet<>();

                    for (var entry : mapDistance.entrySet()) {
                        if (entry.getValue() <= 1)
                            to_keep.add(entry.getKey());
                        else {
                            if (model.removed().contains(entry.getKey()))
                                throw new IllegalStateException(
                                        "The vertex " + entry.getKey()
                                                + " was removed but it is on the remaining graph");
                            to_remove.add(entry.getKey());
                        }
                        if (model.removed().size() >= 0.75 * model.nameToVar().size())
                            break;
                    }
                    setUB(to_remove, 0.0);
                    model.removed().addAll(to_remove);

                    remainingGraph = remainingGraph.subgraph(to_keep);
                    System.out.println("\t- " + to_remove.size() + " variables were fixed;");
                    System.out.println("\t- " + model.removed().size() + " variables out of " + model.nameToVar().size()
                            + " are fixed;");
                }
            }

            elapsedTime = (System.currentTimeMillis() - dinkStart) / 1E3;
            System.out.printf("...Dinkelbach search stopped after %.2fs.\n", elapsedTime);

            if (bestSol.empty)
                throw new RuntimeException("No feasible solution found!");

            if (inst.input_file != null) {
                System.out.printf("\nBest solution found for instance %s:\n", inst.input_file);
            } else {
                System.out.printf("\nBest solution found for instance %dx%dx%d:\n", inst.O, inst.I, inst.A);
            }
            System.out.printf(" - LB = %d, UB = %d;\n", inst.LB, inst.UB);
            System.out.printf(" - %d/%d orders", bestSol.orders.size(), inst.O);
            System.out.printf(" - %d/%d aisles", bestSol.aisles.size(), inst.A);
            System.out.printf(" - %d items;\n", bestSol.wave_items);
            System.out.printf(" - total time: %.2fs;\n", (getCurrentTime(stopWatch)) / 1E3);
            System.out.printf(" - obj: %.2f;\n", bestSol.obj);

            model.cplex().end();
            return bestSol;
        } catch (IloException e) {
            System.out.println(e.getMessage());
        } catch (Exception e) {
            System.out.println("Encountered an error:\n" + e.getMessage());
        }
        assert (false);
        return null;
    }

    private boolean canRemoveAisle(int[] delta, Integer a) {
        var aisleCap = inst.u_ai.get(a);
        for (Map.Entry<Integer, Integer> e : aisleCap.entrySet())
            if (delta[e.getKey()] - e.getValue() < 0)
                return false;
        return true;
    }

    public boolean checkIfHasDemandForOrder(int[] delta, int o) {
        var orderDem = inst.u_oi.get(o);
        for (Map.Entry<Integer, Integer> e : orderDem.entrySet())
            if (delta[e.getKey()] - e.getValue() < 0)
                return false;
        return true;
    }

    public boolean checkIfHasDemandForSwap(int[] delta, int oRem, int oAdd) {
        var orderRem = inst.u_oi.get(oRem);
        var orderAdd = inst.u_oi.get(oAdd);
        for (Map.Entry<Integer, Integer> e : orderRem.entrySet())
            if (delta[e.getKey()] + e.getValue() - orderAdd.getOrDefault(e.getKey(), 0) < 0)
                return false;
        for (Map.Entry<Integer, Integer> e : orderAdd.entrySet())
            if (delta[e.getKey()] - e.getValue() + orderRem.getOrDefault(e.getKey(), 0) < 0)
                return false;
        return true;
    }

    private void localSearchImprove(ChallengeSolution sol) {
        boolean improved = true;
        boolean couldImprove = false;
        double bestObj = sol.obj;

        while (improved) {
            improved = false;
            int[] delta = compute_delta(sol);
            if (delta == null)
                return;

            // --- Try removing redundant aisles ---
            for (Integer a : new HashSet<>(sol.get_aisles())) {
                if (canRemoveAisle(delta, a)) {
                    sol.removeAisle(a);
                    if (!isSolutionFeasible(sol))
                        throw new RuntimeException("Solution should be valid after aisle remove");
                    double obj = computeObjectiveFunction(sol);
                    if (obj <= bestObj)
                        throw new RuntimeException("Solution should be better after aisle remove");
                    bestObj = obj;
                    improved = true;
                    couldImprove = true;
                    sol.obj = obj;
                    delta = compute_delta(sol);
                }
            }

            // --- Try adding profitable orders ---
            for (int o = 0; o < inst.O; o++) {
                if (sol.wave_items + inst.numItemsPerOrder.get(o) > inst.UB || sol.get_orders().contains(o)
                        || inst.invalid_order_nodes.contains(Helpers.oLabel(o)))
                    continue;

                if (checkIfHasDemandForOrder(delta, o)) {
                    sol.addOrder(o, inst.numItemsPerOrder.get(o));
                    if (!isSolutionFeasible(sol))
                        throw new RuntimeException("Solution should be valid after add order");
                    double obj = computeObjectiveFunction(sol);
                    if (obj <= bestObj)
                        throw new RuntimeException("Solution should be better after add order");
                    bestObj = obj;
                    improved = true;
                    couldImprove = true;
                    sol.obj = obj;
                    delta = compute_delta(sol);
                }
            }

            if (improved)
                continue;

            // --- Try swap: remove weak order, add stronger ---
            /*
             * var my_orders = sol.get_orders();
             * for (Integer oRem : my_orders) {
             * if (!sol.get_orders().contains(oRem))
             * continue;
             * int remUnits = inst.numItemsPerOrder.get(oRem);
             *
             * for (Integer oAdd : inst.orderNeighbors.get(oRem)) {
             * if (sol.get_orders().contains(oAdd))
             * continue;
             *
             * int addUnits = inst.numItemsPerOrder.get(oRem);
             * if (addUnits <= remUnits || sol.wave_items + (addUnits - remUnits) > inst.UB)
             * continue;
             *
             * if (checkIfHasDemandForSwap(delta, oRem, oAdd)) {
             * sol.removeOrder(remUnits, remUnits);
             * sol.addOrder(oAdd, addUnits);
             *
             * if (!isSolutionFeasible(sol))
             * throw new RuntimeException("Solution should be valid after swap orders");
             *
             * double obj = computeObjectiveFunction(sol);
             * if (obj <= bestObj)
             * throw new RuntimeException("Solution should be better after swap orders");
             * bestObj = obj;
             * sol.obj = obj;
             * improved = true;
             * couldImprove = true;
             * System.out.printf("\t- LS: swapped order %d -> %d, obj=%.3f\n", oRem, oAdd,
             * obj);
             * delta = compute_delta(sol);
             * break;
             * }
             * }
             * }
             */
        }
        if (couldImprove)
            System.out.printf("\t- LS found a new best solution with obj = %.3f;\n", bestObj);

        // --- Finalize the underlying graph ---
        sol.updateGraph(inst.underlying_graph);

    }

    /*
     * Get the remaining time in seconds
     */
    protected long getRemainingTime(StopWatch stopWatch) {
        var delta = MAX_RUNTIME - stopWatch.getTime(TimeUnit.MILLISECONDS);
        return Math.max(TimeUnit.SECONDS.convert(delta, TimeUnit.MILLISECONDS), 0);
    }

    protected long getCurrentTime(StopWatch stopWatch, TimeUnit unit) {
        return stopWatch.getTime(unit);
    }

    protected long getCurrentTime(StopWatch stopWatch) {
        return getCurrentTime(stopWatch, TimeUnit.MILLISECONDS);
    }

    protected int[] compute_delta(ChallengeSolution sol) {
        Set<Integer> selectedOrders = sol.get_orders();
        Set<Integer> visitedAisles = sol.get_aisles();

        // Basic validity checks
        if (selectedOrders == null || visitedAisles == null ||
                selectedOrders.isEmpty() || visitedAisles.isEmpty()) {
            return null;
        }

        int[] balanceDemandCapacity = new int[inst.I]; // demand from selected orders

        // Sum demand per item from chosen orders
        for (int o : selectedOrders) {
            var orderDem = inst.u_oi.get(o);
            for (Map.Entry<Integer, Integer> e : orderDem.entrySet()) {
                balanceDemandCapacity[e.getKey()] -= e.getValue();
            }
        }

        // Sum capacity per item from chosen aisles
        for (int a : visitedAisles) {
            var aisleCap = inst.u_ai.get(a);
            for (Map.Entry<Integer, Integer> e : aisleCap.entrySet()) {
                balanceDemandCapacity[e.getKey()] += e.getValue();
            }
        }

        return balanceDemandCapacity;
    }

    protected boolean isSolutionFeasible(ChallengeSolution sol) {
        Set<Integer> selectedOrders = sol.get_orders();
        Set<Integer> visitedAisles = sol.get_aisles();

        // Basic validity checks
        if (selectedOrders == null || visitedAisles == null ||
                selectedOrders.isEmpty() || visitedAisles.isEmpty()) {
            return false;
        }

        int[] totalDemand = new int[inst.I]; // demand from selected orders
        int[] totalCapacity = new int[inst.I]; // capacity from selected aisles

        // Sum demand per item from chosen orders
        for (int o : selectedOrders) {
            var orderDem = inst.u_oi.get(o);
            for (Map.Entry<Integer, Integer> e : orderDem.entrySet()) {
                totalDemand[e.getKey()] += e.getValue();
            }
        }

        // Sum capacity per item from chosen aisles
        for (int a : visitedAisles) {
            var aisleCap = inst.u_ai.get(a);
            for (Map.Entry<Integer, Integer> e : aisleCap.entrySet()) {
                totalCapacity[e.getKey()] += e.getValue();
            }
        }

        // Total picked units
        int totalUnits = Arrays.stream(totalDemand).sum();
        if (totalUnits < inst.LB || totalUnits > inst.UB) {
            return false;
        }

        // Demand must not exceed supply for any item
        for (int i = 0; i < inst.I; i++) {
            if (totalDemand[i] > totalCapacity[i]) {
                return false;
            }
        }

        return true;
    }

    protected double computeObjectiveFunction(ChallengeSolution sol) {
        Set<Integer> selectedOrders = sol.get_orders();
        Set<Integer> visitedAisles = sol.get_aisles();

        if (selectedOrders == null || visitedAisles == null ||
                selectedOrders.isEmpty() || visitedAisles.isEmpty()) {
            return 0.0;
        }

        // --- 1. Compute total demand units from selected orders ---
        int totalUnitsPicked = 0;
        for (int o : selectedOrders) {
            Map<Integer, Integer> orderDem = inst.u_oi.get(o);
            for (int demand : orderDem.values()) {
                totalUnitsPicked += demand;
            }
        }

        // --- 2. Compute denominator: number of aisles visited ---
        int numVisitedAisles = visitedAisles.size();

        // --- 3. Objective value ---
        return (double) totalUnitsPicked / numVisitedAisles;
    }
}

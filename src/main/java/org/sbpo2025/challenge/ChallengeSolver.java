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
    private final long MAX_RUNTIME = 600000; // milliseconds; 10 minutes
    Instance inst;

    private record Model(IloCplex cplex, IloNumVar[] Avars, IloNumVar[] Ovars, Map<String, IloNumVar> nameToVar,
                         IloLinearNumExpr waveItemsExpr, ArrayList<String> removed) {
    }

    public ChallengeSolver(Instance instance) {
        this.inst = instance;
    }

    static boolean bestSubSolution(ChallengeSolution sol, int LB) {
        List<Graph> comps = sol.getComponents();
        if (comps.size() <= 1)
            return false;

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
            System.out.printf("- Found a better sub-solution with obj = %.3f;\n", bestLocal);
            sol.restrict_solution(bestComp);
            return true;
        }
        return false;
    }

    // -------------------- Preprocessing: min aisles cover per order ----------
    static int[] minOrdersCover(Instance inst) throws IloException {
        int[] result = new int[inst.O];
        try (IloCplex setModel = new IloCplex()) {
            setModel.setParam(IloCplex.Param.Threads, 8);
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
                        int cap = inst.u_ai.get(a).get(i);
                        if (cap > 0)
                            lhs.addTerm(cap, Avars[a]);
                    }
                    int demand = inst.u_oi.get(o).get(i);
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
    public int getLPSolution(Set<String> vars) {
        try {
            IloCplex cplex = new IloCplex();
            cplex.setOut(new PrintStream(new FileOutputStream("cplex_output_lp.txt")));
            cplex.setParam(IloCplex.Param.Threads, 8);

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

    private void preprocess(Set<String> varInLP) throws IloException {
        System.out.println("Computing invalid orders and isolated vertices...");
        int[] minCovers = minOrdersCover(inst);
        // Mark invalid orders (min cover == 0):
        var invalid = new ArrayList<Integer>();
        for (int o = 0; o < inst.O; o++)
            if (minCovers[o] == 0)
                invalid.add(o);
        int isolated = inst.clear_orders(invalid);
        System.out.printf("\t- %d orders and %d aisles were invalidated;\n", inst.invalid_order_nodes.size(), isolated);

        System.out.println("Running Linear Relaxation...");
        int removedByLP = getLPSolution(varInLP);
        System.out.println("\t- LP-relaxation removed " + removedByLP + " variables;");
    }

    private Model buildModel(HashSet<String> varInLP) throws IloException, FileNotFoundException {
        System.out.println("\nBuilding Dinkelbach model...");
        IloCplex cplex = new IloCplex();
        cplex.setOut(new PrintStream(new FileOutputStream("cplex_output.txt")));
        cplex.setParam(IloCplex.Param.Threads, 7);
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
        var removed = new ArrayList<String>();
        for (String var : nameToVar.keySet()) {
            if (!varInLP.contains(var)) {
                removed.add(var);
                nameToVar.get(var).setUB(0.0);
            }
        }

        return new Model(cplex, Ovars, Avars, nameToVar, waveItemsExpr, removed);
    }

    public ChallengeSolution solve(StopWatch stopWatch) {
        try {
            // ---- Preprocess
            long start = getCurrentTime(stopWatch);
            var varInLP = new HashSet<String>();
            preprocess(varInLP);
            System.out.printf("- Preprocessing completed after %.2fs.\n", (getCurrentTime(stopWatch) - start) / 1E3);

            // ---- Build Dinkelbach model
            start = getCurrentTime(stopWatch);
            Model model = buildModel(varInLP);
            System.out.printf("- Model build completed after %.2fs.\n", (getCurrentTime(stopWatch) - start) / 1E3);

            // --------------- Dinkelbach loop ---------------
            CPLEXSolution bestSol = new CPLEXSolution();
            Graph remainingGraph = inst.underlying_graph; // search space

            final double TOL = 1e-6;
            double OPT_LB = Math.max(inst.LB / (double) Math.max(inst.A, 1), TOL);
            double OPT_UB = inst.trivial_ub();
            double lambda = OPT_LB;

            double elapsedTime = getCurrentTime(stopWatch) / 1E3;
            double totalTime = 60 * 10 - elapsedTime; // 10 minutes minus preprocessing
            double timeTolerance = 10.0;
            double timeLimit = totalTime;

            model.cplex().setParam(IloCplex.DoubleParam.TimeLimit, Math.min(30.0, totalTime));
            model.cplex().setParam(IloCplex.Param.MIP.Pool.Capacity, 2);
            // Upper cutoff -> use as an incumbent cutoff surrogate:
            // cplex.setParam(IloCplex.DoubleParam.MIP.Limits.UpperObjStop, 1.0 / inst.UB);
            // cplex.setParam(IloCplex.Param.MIP.Limits.Solutions, 2);

            System.out.println("\nStarting Dinkelbach search...");
            long dinkStart = System.currentTimeMillis();
            long iterStart = System.currentTimeMillis();
            boolean is_first_iteration = true;

            while (true) {
                double GAP = (OPT_UB - OPT_LB) / Math.max(OPT_LB, TOL);
                System.out.printf("Current interval = [%.3f:%.3f]; gap = %.3f%%; λ = %.2f; ", OPT_LB, OPT_UB, 100 * GAP,
                        lambda);

                // Maximize wave_items - lambda * wave_aisles
                IloLinearNumExpr obj = model.cplex().linearNumExpr();
                //if (!is_first_iteration) {
                obj.add(model.waveItemsExpr());
                //}
                // wave_aisles expression
                IloLinearNumExpr waveAislesExprObj = model.cplex().linearNumExpr();
                for (int a = 0; a < inst.A; a++)
                    waveAislesExprObj.addTerm(-lambda, model.Avars()[a]);
                obj.add(waveAislesExprObj);
                model.cplex().addMaximize(obj);

                boolean solved = model.cplex().solve();
                if (!solved && is_first_iteration) {
                    is_first_iteration = false;
                    for (String var : model.removed()) {
                        IloNumVar v = model.nameToVar().get(var);
                        if (v != null)
                            v.setUB(1.0);
                    }
                    model.removed().clear();
                    continue;
                }
                if (!solved)
                    throw new RuntimeException("No solution found!");

                // Update time limits
                double iterDur = (System.currentTimeMillis() - iterStart) / 1E3;
                iterStart = System.currentTimeMillis();
                timeLimit = Math.max(0, timeLimit - iterDur);
                model.cplex().setParam(IloCplex.DoubleParam.TimeLimit, Math.max(0.0, timeLimit - timeTolerance));

                double dinkObj = model.cplex().getObjValue();
                double elapsed = (System.currentTimeMillis() - dinkStart) / 1E3;
                System.out.printf("dinkelbach obj = %.6f; elapsed %.2fs of %.2fs;\n", dinkObj, elapsed,
                        Math.max(0.0, totalTime - timeTolerance));

                // current objective = wave_items / wave_aisles (values at incumbent)
                double waveItemsVal = model.cplex().getValue(model.waveItemsExpr());
                double waveAislesVal = model.cplex().getValue(waveAislesExprObj) / -lambda;

                double currentObj = waveItemsVal / Math.max(waveAislesVal, 1);
                if (!is_first_iteration && model.removed().isEmpty() && currentObj < -TOL) {
                    throw new IllegalStateException("Negative Objective Function");
                }
                boolean improved = currentObj > bestSol.obj + 1.0 / inst.UB - TOL;
                if (improved) {
                    model.cplex().setParam(IloCplex.DoubleParam.MIP.Limits.LowerObjStop, 1.0 / inst.UB);
                    model.cplex().setParam(IloCplex.Param.MIP.Limits.Solutions, 3);
                    bestSol = new CPLEXSolution(model.cplex(), model.nameToVar(), model.waveItemsExpr(), remainingGraph, false);
                    if (!is_first_iteration
                            && Math.abs(bestSol.wave_items - bestSol.wave_aisles * OPT_LB - dinkObj) >= 1e-3) {
                        throw new IllegalStateException(
                                "Invariant violated: wave_items != wave_aisles * OPT_LB + dinkObj");
                    }
                    System.out.printf("- Found a new best solution with obj = %.3f;\n", bestSol.obj);
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
                        model.cplex().addMIPStart(vars.toArray(new IloNumVar[0]), values, IloCplex.MIPStartEffort.Repair,
                                "MyStart");
                    }
                }
                if (is_first_iteration) {
                    is_first_iteration = false;
                    for (String var : model.removed()) {
                        IloNumVar v = model.nameToVar().get(var);
                        if (v != null)
                            v.setUB(1.0);
                    }
                    model.removed().clear();
                }

                // remove objective to re-set the next iteration
                model.cplex().delete(model.cplex().getObjective());

                if (timeLimit <= timeTolerance)
                    break;

                // Update search interval
                OPT_LB = Math.max(OPT_LB, bestSol.obj);
                lambda = OPT_LB;

                // Local optimum / pruning opportunities
                if (!improved) {
                    System.out.println("- Reached a local optimum!");
                    if (!model.removed().isEmpty()) {
                        // restore
                        Set<String> solution_nodes = new HashSet<>();
                        solution_nodes.addAll(bestSol.aisle_nodes);
                        solution_nodes.addAll(bestSol.order_nodes);
                        var mapDistance = inst.underlying_graph.compute_distance_from_set(solution_nodes);
                        int total = 0;
                        int max = Math.min(5000, (int) Math.ceil(0.6 * model.removed().size()));
                        do {
                            var minDistance = model.removed().stream()
                                    .mapToInt(var -> mapDistance.getOrDefault(var, Integer.MAX_VALUE))
                                    .min().orElse(Integer.MAX_VALUE);
                            var restore = model.removed().stream()
                                    .filter(var -> mapDistance.getOrDefault(var, Integer.MAX_VALUE) <= minDistance)
                                    .toArray(String[]::new);
                            int count = 0;
                            for (String var : restore) {
                                count++;
                                IloNumVar v = model.nameToVar().get(var);
                                if (v != null)
                                    v.setUB(1.0);
                                model.removed().remove(var);
                                if (total + count == max) break;
                            }

                            System.out.printf("- %d fixed variables from distance <= %d were restored;\n", count, minDistance);
                            System.out.println(
                                    "- " + model.removed().size() + " variables out of " + model.nameToVar().size() + " are fixed;");
                            total += count;
                        } while (total < max && !model.removed().isEmpty());
                        var toKeep = inst.underlying_graph.nodes();
                        model.removed().forEach(toKeep::remove);
                        remainingGraph = inst.underlying_graph.subgraph(toKeep);
                    } else {
                        if (model.cplex().getStatus() == IloCplex.Status.Optimal)
                            break;
                        throw new IllegalStateException("Solution should be optimal");
                    }
                } else if (model.nameToVar().size() > 1500) {
                    Set<String> solution_nodes = new HashSet<>();
                    solution_nodes.addAll(bestSol.aisle_nodes);
                    solution_nodes.addAll(bestSol.order_nodes);
                    Map<String, Integer> mapDistance = remainingGraph.compute_distance_from_set(solution_nodes);
                    Set<String> candidate_to_remove = new HashSet<>();
                    Set<String> to_keep = new HashSet<>();

                    for (var entry : mapDistance.entrySet()) {
                        if (entry.getValue() <= 1)
                            to_keep.add(entry.getKey());
                        else {
                            candidate_to_remove.add(entry.getKey());
                        }
                    }
                    Map<String, Integer> already_removed = new HashMap<>();
                    for (String var : model.removed()) {
                        already_removed.put(var, 0);
                    }
                    int removedCount = 0;
                    for (String var : candidate_to_remove) {
                        model.nameToVar().get(var).setUB(0.0);
                        if (!already_removed.containsKey(var)) {
                            removedCount++;
                            model.removed().add(var);
                        }
                    }

                    remainingGraph.subgraph(to_keep);
                    System.out.println("- " + removedCount + " variables were fixed;");
                    System.out.println("- " + model.removed().size() + " variables out of " + model.nameToVar().size() + " are fixed;");
                }
            }

            System.out.printf("...Dinkelbach search stopped after %.2fs.\n",
                    (System.currentTimeMillis() - dinkStart) / 1E3);

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
            System.out.println("Error:\n" + e.getMessage());
        } catch (Exception e) {
            System.out.println("Encountered an error: " + e.getMessage());
        }
        assert (false);
        return null;
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

    protected boolean isSolutionFeasible(ChallengeSolution challengeSolution) {
        Set<Integer> selectedOrders = challengeSolution.get_orders();
        Set<Integer> visitedAisles = challengeSolution.get_aisles();
        if (selectedOrders == null || visitedAisles == null || selectedOrders.isEmpty() || visitedAisles.isEmpty()) {
            return false;
        }

        int[] totalUnitsPicked = new int[inst.I];
        int[] totalUnitsAvailable = new int[inst.I];

        // Calculate total units picked
        /*
         * for (int order : selectedOrders) {
         * for (Map.Entry<Integer, Integer> entry : orders.get(order).entrySet()) {
         * totalUnitsPicked[entry.getKey()] += entry.getValue();
         * }
         * }
         *
         * // Calculate total units available
         * for (int aisle : visitedAisles) {
         * for (Map.Entry<Integer, Integer> entry : aisles.get(aisle).entrySet()) {
         * totalUnitsAvailable[entry.getKey()] += entry.getValue();
         * }
         * }
         */

        // Check if the total units picked are within bounds
        int totalUnits = Arrays.stream(totalUnitsPicked).sum();
        if (totalUnits < inst.LB || totalUnits > inst.UB) {
            return false;
        }

        // Check if the units picked do not exceed the units available
        for (int i = 0; i < inst.I; i++) {
            if (totalUnitsPicked[i] > totalUnitsAvailable[i]) {
                return false;
            }
        }

        return true;
    }

    protected double computeObjectiveFunction(ChallengeSolution challengeSolution) {
        Set<Integer> selectedOrders = challengeSolution.get_orders();
        Set<Integer> visitedAisles = challengeSolution.get_aisles();
        if (selectedOrders == null || visitedAisles == null || selectedOrders.isEmpty() || visitedAisles.isEmpty()) {
            return 0.0;
        }
        int totalUnitsPicked = 0;

        // Calculate total units picked
        /*
         * for (int order : selectedOrders) {
         * totalUnitsPicked += orders.get(order).values().stream()
         * .mapToInt(Integer::intValue)
         * .sum();
         * }
         */

        // Calculate the number of visited aisles
        int numVisitedAisles = visitedAisles.size();

        // Objective function: total units picked / number of visited aisles
        return (double) totalUnitsPicked / numVisitedAisles;
    }
}

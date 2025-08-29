package org.sbpo2025.challenge;

import ilog.concert.IloException;
import ilog.concert.IloNumVar;
import ilog.cplex.IloCplex;

import java.util.HashMap;
import java.util.Map;

class CPLEXSolution extends ChallengeSolution {
    CPLEXSolution(IloCplex cplex,
            Map<String, IloNumVar> nameToVar,
            Integer waveItems,
            Graph instanceGraph,
            boolean empty) throws IloException {
        super(empty ? null : extractSolutionValues(cplex, nameToVar), waveItems, instanceGraph, empty);
    }

    CPLEXSolution(IloCplex cplex,
            Map<String, IloNumVar> nameToVar,
            Integer waveItems,
            Graph instanceGraph) throws IloException {
        this(cplex, nameToVar, waveItems, instanceGraph, false);
    }

    CPLEXSolution() throws IloException {
        this(null, null, 0, null, true);
    }

    private static Map<String, Integer> extractSolutionValues(IloCplex cplex,
            Map<String, IloNumVar> nameToVar)
            throws IloException {
        Map<String, Integer> vals = new HashMap<>();
        for (var e : nameToVar.entrySet()) {
            double v = cplex.getValue(e.getValue());
            vals.put(e.getKey(), (int) Math.round(v));
        }
        return vals;
    }
}

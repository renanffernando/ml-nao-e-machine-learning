package org.sbpo2025.challenge;

import java.util.Set;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;

public class ChallengeSolution {
    boolean empty;

    Map<String, Integer> variables = new HashMap<>();
    Set<String> order_nodes = new HashSet<>();
    List<Integer> orders = new ArrayList<>();
    Set<String> aisle_nodes = new HashSet<>();
    List<Integer> aisles = new ArrayList<>();

    int wave_orders = 0;
    int wave_items = 0;
    int wave_aisles = 0;
    public double obj = Double.NEGATIVE_INFINITY;

    Graph underlying_graph;

    ChallengeSolution(Map<String, Integer> vars, int numItems, Graph instanceGraph, boolean empty) {
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
                orders.add(Helpers.labelNumber(key, 2));
            } else if (c == 'a') {
                aisle_nodes.add(key);
                aisles.add(Helpers.labelNumber(key, 2));
            }
        }
        wave_orders = orders.size();
        wave_aisles = aisles.size();
        obj = wave_items / Math.max(1.0, wave_aisles);
    }

    List<Graph> getComponents() {
        List<Graph> comps = new ArrayList<>();
        for (Set<String> compNodes : underlying_graph.connectedComponents()) {
            comps.add(underlying_graph.subgraph(compNodes));
        }
        return comps;
    }

    void restrict_solution(Graph comp) {
        Set<String> compNodes = comp.getNodes();
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

    public Set<Integer> get_orders() {
        return new LinkedHashSet<>(orders);
    }

    public Set<Integer> get_aisles() {
        return new LinkedHashSet<>(aisles);
    }

    public void addOrder(int o, int demandUnits) {
        String key = Helpers.oLabel(o);
        if (!variables.containsKey(key) || variables.get(key) == 0) {
            variables.put(key, 1);
            order_nodes.add(key);
            orders.add(o);
            wave_orders++;
            wave_items += demandUnits;
            obj = wave_items / Math.max(1.0, wave_aisles);
        }
    }

    public void removeOrder(int o, int demandUnits) {
        String key = Helpers.oLabel(o);
        if (variables.getOrDefault(key, 0) == 1) {
            variables.put(key, 0);
            order_nodes.remove(key);
            orders.remove((Integer) o);
            wave_orders--;
            wave_items -= demandUnits;
            obj = wave_items / Math.max(1.0, wave_aisles);
        }
    }

    public void addAisle(int a) {
        String key = Helpers.aLabel(a);
        if (!variables.containsKey(key) || variables.get(key) == 0) {
            variables.put(key, 1);
            aisle_nodes.add(key);
            aisles.add(a);
            wave_aisles++;
            obj = wave_items / Math.max(1.0, wave_aisles);
        }
    }

    public void removeAisle(int a) {
        String key = Helpers.aLabel(a);
        if (variables.getOrDefault(key, 0) == 1) {
            variables.put(key, 0);
            aisle_nodes.remove(key);
            aisles.remove((Integer) a);
            wave_aisles--;
            obj = wave_items / Math.max(1.0, wave_aisles);
        }
    }

    public void updateGraph(Graph instanceGraph) {
        if (empty) return;
        computeData();
        underlying_graph = instanceGraph.subgraph(union(order_nodes, aisle_nodes));
    }
}
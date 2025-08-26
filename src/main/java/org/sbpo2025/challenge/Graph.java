package org.sbpo2025.challenge;

import java.util.Set;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;

public class Graph {
    Map<String, Set<String>> adj = new HashMap<>();
    Map<String, Integer> nodeWeight = new HashMap<>();
    int graphWeight = 0;

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
        if (!adj.containsKey(n)) return;
        for (String nb : adj.get(n)) {
            adj.get(nb).remove(n);
        }
        adj.remove(n);
        nodeWeight.remove(n);
    }

    Set<String> isolates() {
        Set<String> res = new HashSet<>();
        for (var e : adj.entrySet()) {
            if (e.getValue().isEmpty()) res.add(e.getKey());
        }
        return res;
    }

    Graph subgraph(Set<String> keep) {
        Graph g = new Graph();
        for (String n : keep) {
            if (adj.containsKey(n)) {
                g.addNode(n);
                if (nodeWeight.containsKey(n)) g.setNodeWeight(n, nodeWeight.get(n));
            }
        }
        for (String u : keep) {
            for (String v : neighbors(u)) {
                if (keep.contains(v)) g.addEdge(u, v);
            }
        }
        g.graphWeight = this.graphWeight;
        return g;
    }

    List<Set<String>> connectedComponents() {
        List<Set<String>> comps = new ArrayList<>();
        Set<String> vis = new HashSet<>();
        for (String s : nodes()) {
            if (vis.contains(s)) continue;
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

    Map<String, Integer> compute_distance_from_set(Set<String> node_set) {
        Map<String, Integer> distMap = new HashMap<>();
        Queue<String> queue = new LinkedList<>();

        // Initialize distances
        for (String node : adj.keySet()) {
            distMap.put(node, Integer.MAX_VALUE);
        }

        // Initialize sources
        for (String node : node_set) {
            distMap.put(node, 0);
            queue.add(node);
        }

        // Multi-source BFS
        while (!queue.isEmpty()) {
            String u = queue.poll();
            int distU = distMap.get(u);

            for (String v : adj.getOrDefault(u, new HashSet<String>())) {
                if (distMap.get(v) == Integer.MAX_VALUE) {
                    distMap.put(v, distU + 1);
                    queue.add(v);
                }
            }
        }

        return distMap;
    }
}

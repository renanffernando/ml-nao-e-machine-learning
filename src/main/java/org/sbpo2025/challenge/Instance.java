package org.sbpo2025.challenge;

import java.io.*;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class Instance {
    int O, I, A;
    List<Map<Integer, Integer>> u_oi;
    List<Map<Integer, Integer>> u_ai;

    List<Set<Integer>> order_items = new ArrayList<>(); // per order: items present
    List<Set<Integer>> aisle_items = new ArrayList<>(); // per aisle: items present

    List<Set<Integer>> item_orders; // per item: orders containing it
    List<Set<Integer>> item_aisles; // per item: aisles containing it

    Set<String> invalid_order_nodes = new HashSet<>();
    Set<String> trivial_nodes = new HashSet<>();

    List<Integer> numItemsPerOrder = new ArrayList<>();
    List<List<Integer>> orderNeighbors = new ArrayList<>();

    int LB, UB;

    public double objLB;

    String input_file;
    Graph underlying_graph = new Graph();

    Instance(String inputFilePath) throws IOException {
        this.input_file = inputFilePath;
        BufferedReader br = new BufferedReader(new FileReader(inputFilePath));

        // First line: O I A
        {
            String[] tokens = nextNonEmptyLine(br).trim().split("\\s+");
            O = Integer.parseInt(tokens[0]);
            I = Integer.parseInt(tokens[1]);
            A = Integer.parseInt(tokens[2]);
        }

        item_orders = new ArrayList<>(I);
        item_aisles = new ArrayList<>(I);
        for (int i = 0; i < I; i++) {
            item_orders.add(new HashSet<>());
            item_aisles.add(new HashSet<>());
        }

        // add nodes for bipartite sets
        for (int o = 0; o < O; o++)
            underlying_graph.addNode(Helpers.oLabel(o));
        for (int a = 0; a < A; a++)
            underlying_graph.addNode(Helpers.aLabel(a));

        u_oi = new ArrayList<>(O);
        for (int o = 0; o < O; o++) {
            u_oi.add(new HashMap<>());
            String line = nextNonEmptyLine(br);
            int[] data = parseIntLine(line);

            int sumQty = 0;
            Set<Integer> itemsHere = new HashSet<>();
            for (int p = 1; p + 1 < data.length; p += 2) {
                int idx = data[p];
                int qty = data[p + 1];
                u_oi.get(o).put(idx, qty);
                if (qty > 0) {
                    item_orders.get(idx).add(o);
                    itemsHere.add(idx);
                    sumQty += qty;
                }
            }
            order_items.add(itemsHere);
            underlying_graph.setNodeWeight(Helpers.oLabel(o), sumQty);
            numItemsPerOrder.add(sumQty);
        }

        // For each order o, compute the order that share at least one item with o
        for (int o = 0; o < O; o++) {
            Set<Integer> neighbors = new HashSet<>();
            for (Integer i : u_oi.get(o).keySet())
                neighbors.addAll(item_orders.get(i));
            neighbors.remove(o);
            orderNeighbors.add(neighbors.stream().toList());
        }

        // Read aisles matrix: also sparse pairs
        u_ai = new ArrayList<>(A);
        for (int a = 0; a < A; a++) {
            u_ai.add(new HashMap<>());
            String line = nextNonEmptyLine(br);
            int[] data = parseIntLine(line);
            Set<Integer> itemsHere = new HashSet<>();
            for (int p = 1; p + 1 < data.length; p += 2) {
                int idx = data[p];
                int qty = data[p + 1];
                u_ai.get(a).put(idx, qty);
                if (qty > 0) {
                    item_aisles.get(idx).add(a);
                    itemsHere.add(idx);
                }
            }
            aisle_items.add(itemsHere);
            Set<Integer> seenOrders = new HashSet<>();

            for (int i : itemsHere) {
                for (int o : item_orders.get(i)) {
                    if (!seenOrders.contains(o)) {
                        seenOrders.add(o);
                        underlying_graph.addEdge(Helpers.oLabel(o), Helpers.aLabel(a));
                    }
                }
            }
        }

        {
            String[] tokens = nextNonEmptyLine(br).trim().split("\\s+");
            LB = Integer.parseInt(tokens[0]);
            UB = Integer.parseInt(tokens[1]);
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

    void clear_orders(List<Integer> invalidOrders) {
        for (int o : invalidOrders)
            invalid_order_nodes.add(Helpers.oLabel(o));
        for (String n : invalid_order_nodes)
            underlying_graph.removeNode(n);
        trivial_nodes = underlying_graph.isolates();
        for (String n : trivial_nodes)
            underlying_graph.removeNode(n);
        for (int o : invalidOrders) {
            u_oi.get(o).clear();
            numItemsPerOrder.set(o, 0);
        }
    }

    double trivial_ub() {
        // Sort aisles by "number of items" (sum over items), desc. Compute
        // min(sum_items, UB)/(num_aisles)
        var numAisleItems = new ArrayList<Integer>();
        for (int a = 0; a < A; a++) {
            int tot = 0;
            for (int val : u_ai.get(a).values())
                tot += val;
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
        String[] tokens = line.trim().split("\\s+");
        int[] arr = new int[tokens.length];
        for (int i = 0; i < tokens.length; i++)
            arr[i] = Integer.parseInt(tokens[i]);
        return arr;
    }
}
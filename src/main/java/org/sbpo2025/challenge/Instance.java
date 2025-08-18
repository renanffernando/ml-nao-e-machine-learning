package org.sbpo2025.challenge;

import java.io.BufferedReader;
import java.io.EOFException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class Instance {
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

    Instance(String inputFilePath) {
        try {
            this.input_file = inputFilePath;
            BufferedReader br = new BufferedReader(new FileReader(inputFilePath));

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
                underlying_graph.addNode(Helpers.oLabel(o));
            for (int a = 0; a < A; a++)
                underlying_graph.addNode(Helpers.aLabel(a));

            u_oi = new int[O][I];
            for (int o = 0; o < O; o++) {
                String line = nextNonEmptyLine(br);
                int[] data = parseIntLine(line);

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
                underlying_graph.setNodeWeight(Helpers.oLabel(o), sumQty);
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
                        underlying_graph.addEdge(Helpers.oLabel(o), Helpers.aLabel(a));
                    }
                }
            }

            {
                String[] toks = nextNonEmptyLine(br).trim().split("\\s+");
                LB = Integer.parseInt(toks[0]);
                UB = Integer.parseInt(toks[1]);
            }

            br.close();
        } catch (IOException e) {
            System.out.println("Ocorreu um erro de I/O: " + e.getMessage());
        }
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
            invalid_order_nodes.add(Helpers.oLabel(o));
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
};
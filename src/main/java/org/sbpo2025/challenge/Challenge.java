package org.sbpo2025.challenge;

import org.apache.commons.lang3.time.StopWatch;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

public class Challenge {
    public void writeOutput(ChallengeSolution challengeSolution, String outputFilePath) {
        if (challengeSolution == null) {
            System.err.println("Solution not found");
            return;
        }
        try {
            BufferedWriter writer = new BufferedWriter(new FileWriter(outputFilePath));
            var orders = challengeSolution.get_orders();
            var aisles = challengeSolution.get_aisles();

            // Write the number of orders
            writer.write(String.valueOf(orders.size()));
            writer.newLine();

            // Write each order
            for (int order : orders) {
                writer.write(String.valueOf(order));
                writer.newLine();
            }

            // Write the number of aisles
            writer.write(String.valueOf(aisles.size()));
            writer.newLine();

            // Write each aisle
            for (int aisle : aisles) {
                writer.write(String.valueOf(aisle));
                writer.newLine();
            }

            writer.close();
            System.out.println("Output written to " + outputFilePath);

        } catch (IOException e) {
            System.err.println("Error writing output to " + outputFilePath);
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        // Start the stopwatch to track the running time
        StopWatch stopWatch = StopWatch.createStarted();

        if (args.length != 2) {
            System.out.println("Usage: java -jar target/ChallengeSBPO2025-1.0.jar <inputFilePath> <outputFilePath>");
            return;
        }

        Challenge challenge = new Challenge();
        Instance instance = new Instance(args[0]);
        var challengeSolver = new ChallengeSolver(instance);
        ChallengeSolution challengeSolution = challengeSolver.solve(stopWatch);

        challenge.writeOutput(challengeSolution, args[1]);
    }
}

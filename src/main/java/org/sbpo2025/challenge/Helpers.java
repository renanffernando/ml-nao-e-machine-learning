package org.sbpo2025.challenge;

public class Helpers {
    // --------------------------- Helpers -----------------------------------
    public static String label(String prefix, String key) {
        return prefix + "[" + key + "]";
    }

    public static String oLabel(int o) {
        return label("o", Integer.toString(o));
    }

    public static String aLabel(int a) {
        return label("a", Integer.toString(a));
    }

    public static int labelNumber(String name, int prefixSize) {
        // name like "o[123]": cut off the first 2 chars "o[" and last char "]"
        return Integer.parseInt(name.substring(prefixSize, name.length() - 1));
    }
}
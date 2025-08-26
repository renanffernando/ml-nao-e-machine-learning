package org.sbpo2025.challenge;

public class Pair<F extends Comparable<F>, S extends Comparable<S>> implements Comparable<Pair<F, S>> {

    public F first;
    public S second;

    public Pair(F first, S second) {
        this.first = first;
        this.second = second;
    }

    @Override
    public int compareTo(Pair<F, S> o) {
        int retVal = first.compareTo(o.first);
        if (retVal != 0) return retVal;
        return second.compareTo(o.second);
    }

    @Override
    public String toString() {
        return "(" + first + ", " + second + ")";
    }
}

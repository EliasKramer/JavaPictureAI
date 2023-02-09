package ai.time;

import java.util.Dictionary;
import java.util.Hashtable;

public class TimeHelper {
    private static Dictionary<String, Long> _times = new Hashtable<>();
    public static void start(String name) {
        _times.put(name, System.currentTimeMillis());
    }
    public static long stop(String name) {
        long startTime = _times.get(name);
        long endTime = System.currentTimeMillis();
        long time = endTime - startTime;
        _times.remove(name);
        return time;
    }
    public static String getTimeString(long time) {
        long hours = time / 3600000;
        long minutes = (time % 3600000) / 60000;
        long seconds = (time % 60000) / 1000;
        long milliseconds = time % 1000;

        return String.format("%02dh %02dm %02ds %03dms", hours, minutes, seconds, milliseconds);
    }

}

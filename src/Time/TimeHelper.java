package Time;

public class TimeHelper {
    public static String getTimeString(long time) {
        long hours = time / 3600000;
        long minutes = (time % 3600000) / 60000;
        long seconds = (time % 60000) / 1000;
        long milliseconds = time % 1000;

        return String.format("%02dh %02dm %02ds %03dms", hours, minutes, seconds, milliseconds);
    }
}

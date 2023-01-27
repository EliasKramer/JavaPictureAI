package analyse;

public class AnalyseData {
    public static void saveStringInTxt(String s, String fileName) {
        String path = "analyse/" + fileName + ".txt";

        try {
            java.io.PrintWriter writer = new java.io.PrintWriter(path, "UTF-8");
            writer.println(s);
            writer.close();
        } catch (java.io.IOException e) {
            System.out.println("Error while saving data in " + path);
        }
    }
}

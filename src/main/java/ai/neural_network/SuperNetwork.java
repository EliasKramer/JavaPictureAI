package ai.neural_network;
import ai.time.TimeHelper;
import java.util.Collection;

public class SuperNetwork {
    private final NeuralNetwork[] networks;
    private int[] progressOfNetworks;
    private final int numberOfNetworks;
    private final double noise;
    private int bestIdx = 0;
    private double percentCorrectOfBest = 0.0;
    private static int logThreadIdx = 0;
    public SuperNetwork(
            int numberOfNetworks,
            double noise,
            double randomStartValueRange,
            int numOfInputs,
            int numOfOutputs,
            int numOfHiddenLayers,
            int numOfNeuronsPerHiddenLayer
            ) {

        this.numberOfNetworks = numberOfNetworks;
        this.noise = noise;

        networks = new NeuralNetwork[numberOfNetworks];
        progressOfNetworks = new int[numberOfNetworks];

        for(int i = 0; i < numberOfNetworks; i++)
            networks[i] = new NeuralNetwork(numOfInputs, numOfOutputs, numOfHiddenLayers, numOfNeuronsPerHiddenLayer);

        randomiseWeightsAndBiases(randomStartValueRange);
    }

    private void randomiseWeightsAndBiases(double amount) {
        for(int i = 0; i < numberOfNetworks; i++)
            networks[i].randomiseWeightsAndBiases(amount);
    }

    public void learn(Collection<AiData> data, int numberOfIterations, double stepSize, int numEpochs, int batchSize)
    {
        Thread loggingThread = startLogThread(1000, numEpochs, numberOfIterations);

        for(int iterationIdx = 0; iterationIdx < numberOfIterations; iterationIdx++) {
            //split data into learn and test
            int learnTestDataBorder = (int) (data.size() * 0.9);
            Collection<AiData> learnData = data.stream().toList().subList(0, learnTestDataBorder);
            Collection<AiData> testData = data.stream().toList().subList(learnTestDataBorder, data.size());
            //threads for learning
            Thread[] threads = new Thread[numberOfNetworks];
            //results, that are set after learning (percent correct)
            double[] results = new double[numberOfNetworks];

            //learn in parallel
            for (int i = 0; i < numberOfNetworks; i++) {
                final int idx = i;
                threads[i] = new Thread(() -> {
                    networks[idx].learn(learnData, stepSize, numEpochs, batchSize, this, idx);
                    results[idx] = networks[idx].testOnData(testData, false);
                });
                threads[i].start();
            }

            //wait for all threads to finish
            for (int i = 0; i < numberOfNetworks; i++) {
                try {
                    threads[i].join();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }

            //find best result
            for (int i = 0; i < numberOfNetworks; i++) {
                if (results[i] > percentCorrectOfBest) {
                    percentCorrectOfBest = results[i];
                    bestIdx = i;
                }
            }

            //apply noise to all networks except the best one
            System.out.println("Best result: " + percentCorrectOfBest);
            for (int i = 0; i < numberOfNetworks; i++) {
                if (i != bestIdx) {
                    networks[i] = networks[bestIdx].getCopy();
                    networks[i].applyNoise(noise);
                }
            }
        }
        //stopLogThread(loggingThread);
    }
    public NeuralNetwork getBestNetwork()
    {
        return networks[bestIdx];
    }
    private Thread startLogThread(long interval, int numEpochs, int iterations)
    {
        final int maxSumOfEpochs = numberOfNetworks * numEpochs * iterations;
        Thread t = new Thread(() -> {
            long startTime = System.currentTimeMillis();
            long lastUpdate = startTime;
            int sum = 0;
            while(sum < maxSumOfEpochs)
            {
                if(System.currentTimeMillis() - lastUpdate > interval)
                {
                    lastUpdate = System.currentTimeMillis();
                    sum = 0;
                    for(int i = 0; i < numberOfNetworks; i++)
                    {
                        sum += progressOfNetworks[i];
                    }
                    double percentDone = (double) sum / maxSumOfEpochs * 100;
                    int currIteration = sum / (numEpochs * numberOfNetworks) + 1;
                    System.out.println(
                            Thread.currentThread().getName() + " " +
                            "Done: " +
                                    String.format("%.2f", percentDone) + "%" +
                                    " | Time passed: " +
                                    TimeHelper.getTimeString(((System.currentTimeMillis() - startTime))) +
                                    " | Time remaining: " +
                                    TimeHelper.getTimeString((long)
                                            ((System.currentTimeMillis() - startTime) /
                                                    percentDone * (100 - percentDone))) +
                                    " | Best Neural Network : " + percentCorrectOfBest + "%" +
                                    " | iteration " + currIteration + "/" + iterations
                    );

                }
            }
        });
        logThreadIdx++;
        t.setName("T_" + logThreadIdx);
        t.start();

        return t;
    }
    private void stopLogThread(Thread t)
    {
        for(int i = 0; i < numberOfNetworks; i++)
            progressOfNetworks[i] = 0;
        //wait for thread to finish after setting loggingThreadShouldRun to false
        try {
            t.join();
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
    }
    public void updateProgress(int networkId) {
        progressOfNetworks[networkId] ++;
    }
}

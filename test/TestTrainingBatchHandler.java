import ai.neural_network.TrainingBatchHandler;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.LinkedList;
import java.util.List;

public class TestTrainingBatchHandler {
    @Test
    void testConstructor() {
        try{
            TrainingBatchHandler<Integer> tbh = new TrainingBatchHandler<>(null);
        }
        catch (IllegalArgumentException e){
            Assertions.assertTrue(true);
        }
        try{
            TrainingBatchHandler<Integer> tbh = new TrainingBatchHandler<>(new LinkedList<>());
        }
        catch (IllegalArgumentException e){
            Assertions.assertTrue(true);
        }
    }
    @Test
    void testGetOne()
    {
        LinkedList<Integer> list = new LinkedList<>();
        list.add(1);
        TrainingBatchHandler<Integer> tbh = new TrainingBatchHandler<>(list);
        List<Integer> batch = tbh.getNewRandomBatch(1);
        Assertions.assertEquals(1, batch.size());
        Assertions.assertTrue(list.contains(batch.get(0)));
    }
    @Test
    void testGetTwo()
    {
        LinkedList<Integer> list = new LinkedList<>();
        list.add(1);
        list.add(2);
        TrainingBatchHandler<Integer> tbh = new TrainingBatchHandler<>(list);
        List<Integer> batch = tbh.getNewRandomBatch(2);
        Assertions.assertEquals(2, batch.size());
        Assertions.assertTrue(list.contains(batch.get(0)));
        Assertions.assertTrue(list.contains(batch.get(1)));
    }
    @Test
    void testGetMultiple()
    {
        LinkedList<Integer> list = new LinkedList<>();
        list.add(1);
        list.add(2);
        list.add(3);
        list.add(4);
        list.add(5);
        TrainingBatchHandler<Integer> tbh = new TrainingBatchHandler<>(list);
        List<Integer> batch = tbh.getNewRandomBatch(3);
        Assertions.assertEquals(3, batch.size());

        list.remove(batch.get(0));
        list.remove(batch.get(1));
        list.remove(batch.get(2));

        Assertions.assertEquals(2, list.size());

        List<Integer> secondBatch = tbh.getNewRandomBatch(2);

        Assertions.assertEquals(2, secondBatch.size());

        list.remove(secondBatch.get(0));
        list.remove(secondBatch.get(1));

        Assertions.assertEquals(0, list.size());
    }
}

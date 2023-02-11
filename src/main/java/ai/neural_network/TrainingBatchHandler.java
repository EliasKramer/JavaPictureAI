package ai.neural_network;

import java.util.Collection;
import java.util.Collections;
import java.util.List;

public class TrainingBatchHandler<T> {
    private final Collection<T> _data;
    private int _currentIdx;
    private int _shuffleCount;
    public TrainingBatchHandler(Collection<T> data) {
        if(data == null)
            throw new IllegalArgumentException("Data cannot be null.");
        if(data.size() == 0)
            throw new IllegalArgumentException("Data cannot be empty.");
        _data = data;
        shuffleData();
    }

    public List<T> getNewRandomBatch(int size) {
        if(size > _data.size() || size < 0)
            throw new IllegalArgumentException("Batch size cannot be handled. Either too big or too small.");

        if(_currentIdx + size > _data.size())
            shuffleData();

        List<T> retVal = _data.stream().skip(_currentIdx).limit(size).toList();
        _currentIdx += size;
        return retVal;
    }

    private void shuffleData() {
        Collections.shuffle((java.util.List<T>) _data);
        _currentIdx = 0;
        _shuffleCount++;
    }
}
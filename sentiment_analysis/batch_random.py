def generate_batch_data_random(x, y, batch_size):
    ylen = len(y)
    loopcount = (ylen + batch_size - 1) // batch_size
    while (True):
        for i in range(loopcount):
            lis = []
            for e in x:
                lis.append(e[i * batch_size:min((i + 1) * batch_size, ylen)])
            yield lis, y[i * batch_size:min((i + 1) * batch_size, ylen)]
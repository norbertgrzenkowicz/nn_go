FROM golang:1.22.6-alpine

RUN apk add --no-cache curl

WORKDIR /app

# I started to get permission error on getting these files from og mnist website
RUN curl -o train-images-idx3-ubyte.gz https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz \
    && curl -o train-labels-idx1-ubyte.gz https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz \
    && curl -o t10k-images-idx3-ubyte.gz https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz \
    && curl -o t10k-labels-idx1-ubyte.gz https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz

RUN gunzip train-images-idx3-ubyte.gz \
    && gunzip train-labels-idx1-ubyte.gz \
    && gunzip t10k-images-idx3-ubyte.gz \
    && gunzip t10k-labels-idx1-ubyte.gz

COPY . .

RUN go mod tidy

RUN go build -o nn .

CMD ["./nn"]

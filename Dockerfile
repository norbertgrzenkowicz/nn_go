FROM golang:1.22.6-alpine

RUN apk add --no-cache curl

WORKDIR /app

# I started to get permission error on getting these files from og mnist website
RUN curl -o mnist/train-images-idx3-ubyte.gz http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz \
    && curl -o mnist/train-labels-idx1-ubyte.gz http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz \
    && curl -o mnist/t10k-images-idx3-ubyte.gz http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz \
    && curl -o mnist/t10k-labels-idx1-ubyte.gz http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

RUN gunzip train-images-idx3-ubyte.gz \
    && gunzip train-labels-idx1-ubyte.gz \
    && gunzip t10k-images-idx3-ubyte.gz \
    && gunzip t10k-labels-idx1-ubyte.gz

COPY . .

RUN go mod tidy

RUN go build -o main ./src/

CMD ["./main"]

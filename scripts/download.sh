FILENAME="cifar-10-python.tar.gz"
URL="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

cd ..
mkdir data
cd data
curl $URL --output $FILENAME
tar -xvzf $FILENAME
rm $FILENAME
# Simple GAN Example

> Quite **simple and clean** code to explain how GAN works.

Thanks to [MorvanZhou](https://github.com/MorvanZhou) and his [excellent video(in Chinese)](https://www.youtube.com/watch?v=EPAIUW_A4sU)

# Define generator

```
def generator():
    G = Sequential()
    G.add(Dense(128, input_shape=(DRAW_SIZE,), activation='relu'))
    G.add(Dense(DRAW_SIZE))
    return G
```

# Define discriminator

```
def discriminator():
    D = Sequential()
    D.add(Dense(128, input_shape=(DRAW_SIZE,), activation='relu'))
    D.add(Dense(1, activation='sigmoid'))
    return D
```

# Define GAN

```
def GAN(G, D):
    model = Sequential()
    model.add(G)
    D.trainable = False # Yes, don't evolve when judge generator
    model.add(D)
    return model
```

# Train(pseudo-code)

```
def train(G, D, M):
    # Step 1, Train discriminator
    Compose X1, y1. Half true sample, half generated
    D.trainable = True
    D.train_on_batch(X1, y1)

    # Step 2, Train GAN(mainly on generator, because discriminator is not trainable for now)
    Compose X2, y2, all generated. (So, y2 is all 0)
    D.trainable = False
    M.train_on_batch(X2, y2)

for i in range(100000):
    train(G, D, M)
    Print sth. periodically
```

It's just so simple. Star/Fork if it helps or you wanna a try for yourself.

import matplotlib.pyplot as plt

def plot_losses(losses, title="Loss", xlabel="epoch"):
    plt.figure(figsize=(6,4))
    plt.plot(losses)
    plt.xlabel(xlabel)
    plt.ylabel("loss")
    plt.title(title)
    plt.grid(True)
    plt.show()

def topk_table(movie_titles, scores, k=10):
    idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
    return [(i+1, movie_titles[i], float(scores[i])) for i in idx]

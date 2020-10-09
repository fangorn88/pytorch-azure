import matplotlib.pyplot as plt


def plot_losses(train_loss_history, valid_loss_history, best_valid_score, burn_in: int = 0):
    plt.figure(figsize=(16, 5))
    plt.plot(train_loss_history[burn_in:], label="train")
    plt.plot(valid_loss_history[burn_in:], label="valid")

    epoch_best_valid_score = valid_loss_history.index(best_valid_score)
    plt.axvline(x=epoch_best_valid_score,  c="g", label=f"best_epoch: {epoch_best_valid_score}")
    plt.ylabel("loss function")
    plt.xlabel("number of epochs")
    plt.legend()
    plt.grid()
    plt.show()

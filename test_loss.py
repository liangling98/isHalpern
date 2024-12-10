import matplotlib.pyplot as plt
import numpy as np
from function import isHI_WDRO


sample_size = 100
X = np.random.normal(0, 1, (sample_size, 3)) 
mean = np.average(X, axis=0)

Halpern_method = isHI_WDRO(3)
Halpern_method.update(config={'eps': 0.01})

#plot the figure of test loss
plt.figure(figsize=(8, 6), dpi=300)
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'  
plt.rcParams['ytick.color'] = 'black'     
plt.rcParams['axes.labelcolor'] = 'black'  
plt.rcParams['text.color'] = 'black'       
plt.rcParams['legend.edgecolor'] = 'black' 

all_losses = []

for i in range(10):
    theta = Halpern_method.fit(X)
    temp = Halpern_method.worst_distribution()
    loss = []
    for control in Halpern_method.decision:
        loss.append(control @ control/2 - np.sum(temp, axis=0)@control/sample_size - np.sum(np.exp(-np.sum(np.power(temp, 2), axis=1)))/sample_size/2 - np.sum(np.exp(-np.sum(np.power(temp, 2), 1)))/sample_size/2)
    all_losses.append(loss)


# Align losses (padding with NaNs for unequal lengths)
max_len = max(len(loss) for loss in all_losses)
padded_losses = [np.pad(loss, (0, max_len - len(loss)), constant_values=np.nan) for loss in all_losses]

# Calculate statistics
padded_losses = np.array(padded_losses)
mean_loss = np.nanmean(padded_losses, axis=0)
std_loss = np.nanstd(padded_losses, axis=0)
lower_bound = mean_loss - std_loss
upper_bound = mean_loss + std_loss

iterations = range(1, len(mean_loss) + 1)
plt.plot(iterations, mean_loss, label="isHI", color="dodgerblue", linewidth = 1.5)
plt.fill_between(iterations, lower_bound, upper_bound, color="dodgerblue", alpha=0.2, label="Loss Range (±1 std)")

plt.xlabel("Number of Iterations")
plt.ylabel("Test Losses")
plt.title("Test loss of isHI solution")
plt.legend()
plt.grid(True, color="gray", alpha=0.2)
plt.savefig("./test_loss.png")
plt.show()
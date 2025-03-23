from matplotlib import pyplot as plt
import seaborn as sns

def linear_reg_plot(df, xname, yname):
    plt.hexbin(df[xname], df[yname], gridsize=20, cmap='Blues', mincnt=1, edgecolor="#B0B0B0")
    plt.colorbar()
    sns.regplot(x=xname, y=yname, data=df, scatter=False, color='red')
    plt.show()
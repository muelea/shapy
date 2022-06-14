
import os 
import joblib
import matplotlib.pyplot as plt
from attributes.utils.constants import ATTRIBUTE_NAMES

def plot_ratings(ratings, betas, gender, outdir):

    ATR = ATTRIBUTE_NAMES[gender]
    os.makedirs('{}/{}'.format(outdir, gender), exist_ok=True)
    for idx in range(ratings.shape[1]):
        for beta_idx in range(betas.shape[1]):
            aname = ATR[idx]
            plt.plot(ratings[:,idx], betas[:,beta_idx], '.')
            plt.savefig('{}/{}/{}_{}'.format(outdir, gender, aname, beta_idx))
            plt.close()

if __name__ == "__main__":

    NUM_BETAS = 10
    MODEL = 'smplx'
    OUTDIR = '../out/plots_attribute_betas'

    for gender in ['male', 'female']:
        data = joblib.load(f'../data/dbs/caesar_{gender}_train.pt')
        ratings = data['ratings']
        betas = data[f'betas_{MODEL}_{gender}'][:, :NUM_BETAS]
        plot_ratings(ratings, betas, gender, OUTDIR)
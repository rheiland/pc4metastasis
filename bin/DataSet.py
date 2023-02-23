import os  # Randy
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
import arviz as az

from MCMC import *
from ModelSelection import *

# Result MCMC Breast
MCMC_Burn_in = 100
# folder = "MCMC_Files/Model_Candidates/"
folder = "data/Model_Candidates/"   # Randy - but overridden in Loading_ module anyway


# Breast
NumPar_breast, MLE_Par_breast, MLE_Solution_breast , Median_Par_breast, Mean_Par_breast, LL_value_breast, RSS_value_breast, FlatSamples_breast, Samples_Chains_breast, Samples_Chains_loglik_breast = Loading_MCMC_breast(folder, burn_in = MCMC_Burn_in)
dataSize_breast = len(obs_data.DsRedBreastMean)+len(obs_data.GFPBreastMean)
AIC_breast, AICc_breast, BIC_breast =  Model_Selection(MLE_Par_breast, LL_value_breast, dataSize_breast, constraints_breast) # Using log likelihood
w_AIC_breast, Delta_AIC_breast = weight_InfCriterion(AIC_breast)
ER_AIC_breast = evidence_ratio(Delta_AIC_breast)
w_AICc_breast, Delta_AICc_breast = weight_InfCriterion(AICc_breast)
ER_AICc_breast = evidence_ratio(Delta_AICc_breast)
w_BIC_breast, Delta_BIC_breast = weight_InfCriterion(BIC_breast)
ER_BIC_breast = evidence_ratio(Delta_BIC_breast)
df_ModelSelection_Breast = pd.DataFrame(data={'ModelID': range(1,len(constraints_breast)+1), 'Num Parameters': NumPar_breast, 'MLE Parameter': MLE_Par_breast, 'Mean Parameter': Mean_Par_breast, 'Median Parameter': Median_Par_breast, 'MLE Smooth Solution': MLE_Solution_breast, 'Max Likelihood': np.exp(LL_value_breast), 'RSS': RSS_value_breast, 'Delta AIC': Delta_AIC_breast, 'weight AIC': w_AIC_breast, 'Evidence ratio AIC': ER_AIC_breast, 'Delta AICc': Delta_AICc_breast, 'weight AICc': w_AICc_breast, 'Evidence ratio AICc': ER_AICc_breast, 'Delta BIC': Delta_BIC_breast, 'weight BIC': w_BIC_breast, 'Evidence ratio BIC': ER_BIC_breast})
# Bloodstream and Lung
modelID_breast = 14 # BestModel
theta_breast = df_ModelSelection_Breast.loc[modelID_breast-1,'MLE Parameter'][0]
NumPar_mets, MLE_Par_mets, MLE_Solution_mets , Median_Par_mets, Mean_Par_mets, LL_value_mets, RSS_value_mets, FlatSamples_mets, Samples_Chains_mets, Samples_Chains_loglik_mets = Loading_MCMC_mets(folder,theta_breast, modelID_breast, burn_in = MCMC_Burn_in)
dataSize_mets = len(obs_data.DsRedBloodStMean)+len(obs_data.GFPBloodStMean)+len(obs_data.DsRedLungMean)+len(obs_data.GFPLungMean)
AIC_mets, AICc_mets, BIC_mets =  Model_Selection(MLE_Par_mets, LL_value_mets, dataSize_mets, constraints_mets) # Using log likelihood
w_AIC_mets, Delta_AIC_mets = weight_InfCriterion(AIC_mets)
ER_AIC_mets = evidence_ratio(Delta_AIC_mets)
w_AICc_mets, Delta_AICc_mets = weight_InfCriterion(AICc_mets)
ER_AICc_mets = evidence_ratio(Delta_AICc_mets)
w_BIC_mets, Delta_BIC_mets = weight_InfCriterion(BIC_mets)
ER_BIC_mets = evidence_ratio(Delta_BIC_mets)
df_ModelSelection_Mets = pd.DataFrame(data={'ModelID': range(1,len(constraints_mets)+1), 'Num Parameters': NumPar_mets, 'MLE Parameter': MLE_Par_mets, 'Mean Parameter': Mean_Par_mets, 'Median Parameter': Median_Par_mets, 'MLE Smooth Solution': MLE_Solution_mets, 'Max Likelihood': np.exp(LL_value_mets), 'RSS': RSS_value_mets, 'Delta AIC': Delta_AIC_mets, 'weight AIC': w_AIC_mets, 'Evidence ratio AIC': ER_AIC_mets, 'Delta AICc': Delta_AICc_mets, 'weight AICc': w_AICc_mets, 'Evidence ratio AICc': ER_AICc_mets, 'Delta BIC': Delta_BIC_mets, 'weight BIC': w_BIC_mets, 'Evidence ratio BIC': ER_BIC_mets})

class ScalarFormatterClass(ScalarFormatter):
   def _set_format(self):
      self.format = "%1.1f"

def Plot_curves_breast(ModelID=None, FontSize=None, FileName=None):
    fig, ax1 = plt.subplots(1, 1,figsize=(6, 4))
    if (ModelID):
        output = df_ModelSelection_Breast.loc[ModelID-1,'MLE Smooth Solution']
        ax1.plot(smooth_time,output[:,0],color="red")
        ax1.plot(smooth_time,output[:,1],color="green")
    ax1.scatter(obs_data.obs_time,obs_data.DsRedBreastMean,color="red")
    ax1.scatter(obs_data.obs_time,obs_data.GFPBreastMean,color="green")
    ax1.errorbar(obs_data.obs_time[1:],obs_data.DsRedBreastMean[1:],yerr=obs_data.DsRedBreastStd[1:],ls='none',ecolor='red',capsize=10)
    ax1.errorbar(obs_data.obs_time[1:],obs_data.GFPBreastMean[1:],yerr=obs_data.GFPBreastStd[1:],ls='none',ecolor='green',capsize=10)
    ax1.ticklabel_format(axis='y', style='sci', scilimits=(-1,1), useMathText=True)
    ax1.set(xlabel = "Time (days)", ylabel = "Number of cells", title = "Breast")
    if FontSize:
        for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label, ax1.yaxis.offsetText] + ax1.get_xticklabels() + ax1.get_yticklabels()):
            item.set_fontsize(FontSize)
    fig.tight_layout()
    if FileName: plt.savefig(FileName)
    else: plt.show()

def Plot_curves_mets(ModelID=None, FontSize=None, FileName=None):
    fig, (ax1,ax2,ax3) = plt.subplots(1, 3,figsize=(18, 4))
    if (ModelID):
        output = df_ModelSelection_Mets.loc[ModelID-1,'MLE Smooth Solution']
        ax1.plot(smooth_time,output[:,0],color="red")
        ax1.plot(smooth_time,output[:,1],color="green")
        ax2.plot(smooth_time,output[:,2],color="red")
        ax2.plot(smooth_time,output[:,3],color="green")
        ax3.plot(smooth_time,output[:,4],color="red")
        ax3.plot(smooth_time,output[:,5],color="green")
    ax1.scatter(obs_data.obs_time,obs_data.DsRedBreastMean,color="red")
    ax1.scatter(obs_data.obs_time,obs_data.GFPBreastMean,color="green")
    ax1.errorbar(obs_data.obs_time[1:],obs_data.DsRedBreastMean[1:],yerr=obs_data.DsRedBreastStd[1:],ls='none',ecolor='red',capsize=10)
    ax1.errorbar(obs_data.obs_time[1:],obs_data.GFPBreastMean[1:],yerr=obs_data.GFPBreastStd[1:],ls='none',ecolor='green',capsize=10)
    # ax1.ticklabel_format(axis='y', style='sci', scilimits=(-1,1), useMathText=True)
    ax2.scatter([obs_data.obs_time[0], obs_data.obs_time[-1]], obs_data.DsRedBloodStMean, color='red')
    ax2.scatter([obs_data.obs_time[0], obs_data.obs_time[-1]], obs_data.GFPBloodStMean, color='green')
    ax2.errorbar(obs_data.obs_time[-1],obs_data.DsRedBloodStMean[-1], yerr=obs_data.DsRedBloodStStd[-1], ls='none',ecolor='red',capsize=10)
    ax2.errorbar(obs_data.obs_time[-1], obs_data.GFPBloodStMean[-1], yerr=obs_data.GFPBloodStStd[-1], ls='none',ecolor='green',capsize=10)
    # ax2.ticklabel_format(axis='y', style='sci', scilimits=(-1,1), useMathText=True)
    ax3.scatter(obs_data.obs_time,obs_data.DsRedLungMean,color="red")
    ax3.scatter(obs_data.obs_time,obs_data.GFPLungMean,color="green")
    ax3.errorbar(obs_data.obs_time[1:],obs_data.DsRedLungMean[1:],yerr=obs_data.DsRedLungStd[1:],ls='none',ecolor='red',capsize=10)
    ax3.errorbar(obs_data.obs_time[1:],obs_data.GFPLungMean[1:],yerr=obs_data.GFPLungStd[1:],ls='none',ecolor='green',capsize=10)
    # ax3.ticklabel_format(axis='y', style='sci', scilimits=(-1,1), useMathText=True)
    ax1.set(xlabel = "Time (days)", ylabel = "Number of cells", title = "Breast tumor")
    ax2.set(xlabel = "Time (days)", ylabel = "Number of cells", title = "Bloodstream")
    ax3.set(xlabel = "Time (days)", ylabel = "Number of cells", title = "Lung")#, yscale='symlog')
    yScalarFormatter = ScalarFormatterClass(useMathText=True)
    yScalarFormatter.set_powerlimits((-1,1))
    ax1.yaxis.set_major_formatter(ScalarFormatterClass(useMathText=True))
    ax2.yaxis.set_major_formatter(yScalarFormatter)
    ax3.yaxis.set_major_formatter(yScalarFormatter)
    if FontSize:
        for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label, ax1.yaxis.offsetText] + ax1.get_xticklabels() + ax1.get_yticklabels()):
            item.set_fontsize(FontSize)
        for item in ([ax2.title, ax2.xaxis.label, ax2.yaxis.label, ax2.yaxis.offsetText] + ax2.get_xticklabels() + ax2.get_yticklabels()):
            item.set_fontsize(FontSize)
        for item in ([ax3.title, ax3.xaxis.label, ax3.yaxis.label, ax3.yaxis.offsetText] + ax3.get_xticklabels() + ax3.get_yticklabels()):
            item.set_fontsize(FontSize)
    fig.tight_layout()
    if FileName: plt.savefig(FileName)
    else: plt.show()

def ModelsStructures_breast():
    caption_size = 'h4'
    text1 = widgets.HTMLMath( value=r"Let $[B_r]$ and $[B_g]$ the number of red and green cells in breast. The variation of cell population is given by:")
    text2 = widgets.HTMLMath( value=r"$$\begin{align} \frac{d[B_r]}{dt} &=& \mathcal{G}_{[B_r]} - \lambda [B_r] - \mathcal{I}_{[B_r]}\\ \frac{d[B_g]}{dt} &=& \mathcal{G}_{[B_g]} + \lambda [B_r] - \mathcal{I}_{[B_g]}\end{align}$$")
    title3 = widgets.HTML(value='<{size}> Subject to mathematical assumptions: </{size}>'.format(size=caption_size))
    text3 = widgets.HTMLMath( value=r"$$\begin{align} \fbox{EG}: \mathcal{G}_{[i]} &=& \alpha_{[i]}[i] \quad or \quad \fbox{LG}: \mathcal{G}_{[i]}=\alpha_{[i]}\left( 1 - \frac{[i]}{K_{[i]}} \right), \quad i=\{B_r,B_g\}\\ \fbox{NDI}: \mathcal{I}_{[i]} &=& \beta_{[i]}[i] \quad or \quad \fbox{DI}: \mathcal{I}_{[i]}=\beta_{[i]} \mathcal{H}(t-\tau)[i],  \quad i=\{B_r,B_g \} \end{align}$$")
    text4 = widgets.HTMLMath( value=r"where $\fbox{EG}$ and $\fbox{LG}$ are exponential and logistic growth, $\fbox{DI}$ and $\fbox{NDI}$ are delay and no delay intravasation, repectively, and $\mathcal{H}(x) = \begin{cases} 1 & \text{if } x\geq0\\ 0 & \text{else} \end{cases}$ is the Heaviside function.")
    title5 = widgets.HTML(value='<{size}> Subject to parameters assumptions: </{size}>'.format(size=caption_size))
    text5 = widgets.HTMLMath( value=r"$$\begin{align} \fbox{H1}&: \alpha_{[B_r]} = \alpha_{[B_g]} &\quad \mbox{The growth rate of red and green cells are equal on breat tissue.}\\ \fbox{H2}&: \beta_{[B_r]} = \beta_{[B_g]} &\quad \mbox{The intravasation rate of red and green cells are equal on breat tissue.} \end{align}$$")
    display(text1,text2,title3,text3,text4,title5,text5)

def ModelsStructures_mets():
    caption_size = 'h4'
    text1 = widgets.HTMLMath( value=r"Let $[B_r]$, $[B_g]$, $[C_r]$, $[C_g]$, $[L_r]$, and $[L_g]$ the number of red and green cells in breast tumor, bloodstream, and lung. The variation of cell population is given by:")
    text2 = widgets.HTMLMath( value=r"$$\begin{align} \color{gray}{\frac{d[B_r]}{dt}} &\color{gray}{= \mathcal{G}_{[B_r]} - \lambda [B_r] - \mathcal{I}_{[B_r]}}\\ \color{gray}{\frac{d[B_g]}{dt}} &\color{gray}{= \mathcal{G}_{[B_g]} + \lambda [B_r] - \mathcal{I}_{[B_g]}}\\ \frac{d[C_r]}{dt} &= \mathcal{I}_{[B_r]} - \phi_{r}[C_r]\\ \frac{d[C_g]}{dt} &= \mathcal{I}_{[B_g]} - \phi_{g}[C_g]\\ \frac{d[L_r]}{dt} &= f_r \phi_{r}[C_r] + \mathcal{G}_{[L_r]}\\ \frac{d[L_g]}{dt} &= f_g \phi_{g}[C_g] + \mathcal{G}_{[L_g]} \end{align}$$")
    title3 = widgets.HTML(value='<{size}>Subject to mathematical assumptions: </{size}>'.format(size=caption_size))
    text3 = widgets.HTMLMath( value=r"$$\begin{align} \fbox{EG}: \mathcal{G}_{[i]} &=& \alpha_{[i]}[i] \quad or \quad \fbox{LG}: \mathcal{G}_{[i]}=\alpha_{[i]}\left( 1 - \frac{[i]}{K_{[i]}} \right), \quad i=\{L_r,L_g\} \end{align}$$")
    text4 = widgets.HTMLMath( value=r"where $\fbox{EG}$ and $\fbox{LG}$ are exponential and logistic growth, repectively")
    title5 = widgets.HTML(value='<{size}> Subject to parameters assumptions: </{size}>'.format(size=caption_size))
    text5 = widgets.HTMLMath( value=r"$$\begin{align} \fbox{H3}&: \phi_{r} = 3*\phi_{g} &\quad \mbox{The green circulating tumor cells (CTCs) are three times more resistent than red CTCs.}\\ \fbox{H4}&: f_r = f_g &\quad \mbox{The extravastion fraction of red and green cells to lung are equal.}\\ \fbox{H5}&: \alpha_{[L_r]} = \alpha_{[L_g]} &\quad \mbox{The growth rate of red and green cells are equal on lung.} \end{align}$$")
    display(text1,text2,title3,text3,text4,title5,text5)

def Grid_Constraints(Constraints):
    Ncolummns = 3
    Nrows = len(Constraints)//Ncolummns + (len(Constraints)%Ncolummns > 0)
    grid = GridspecLayout(Nrows, Ncolummns, width='auto', height='auto')
    for i in range(Nrows):
        for j in range(Ncolummns):
            idx_Model = j*Nrows + i
            if (idx_Model >= len(Constraints)): break
            value = ""
            if (len(Constraints[idx_Model]) == 0):
                value = "No constraint"
            else:
                value = '\n'.join(Constraints[idx_Model])
            grid[i,j] =  widgets.Textarea( value=value, description='Model %d:'%(idx_Model+1), disabled=True, layout=widgets.Layout(height="100px", width="auto"))
    return grid

def data_frame_parameter_result(ID_model, param_value, constraints_breast, parameters_breast_df):
    idx = Apply_Constraints(constraints_breast[ID_model-1],parameters_breast_df)
    return pd.DataFrame(data={'Parameters': parameters_breast_df["Parameter"].iloc[idx], 'Value': param_value, 'Unit': parameters_breast_df["Unit"].iloc[idx], 'Description': parameters_breast_df["Description"].iloc[idx]})

def TabResult_breast(ModelID):
    def setup_ui(df):
        out = widgets.Output()
        with out:
            display(df)
        return out
    value = '\n'.join(constraints_breast[ModelID-1])
    A = setup_ui(data_frame_parameter_result(ModelID, df_ModelSelection_Breast.loc[ModelID-1,'MLE Parameter'][0], constraints_breast, parameters_breast_df))
    B = widgets.Textarea( value=value, description='Constraints:', disabled=True, layout=widgets.Layout(height="200px", width="auto"))
    display(widgets.HBox([A,B]))

def TabResult_mets(ModelID):
    def setup_ui(df):
        out = widgets.Output()
        with out:
            display(df)
        return out
    value = '\n'.join(constraints_mets[ModelID-1])
    A = setup_ui(data_frame_parameter_result(ModelID, df_ModelSelection_Mets.loc[ModelID-1,'MLE Parameter'][0], constraints_mets, parameters_mets_df))
    B = widgets.Textarea( value=value, description='Constraints:', disabled=True, layout=widgets.Layout(height="200px", width="auto"))
    display(widgets.HBox([A,B]))

def Plot_MLE_Solution_Breast(ModelID):
    Plot_curves_breast(ModelID)
    print(f"Number of parameter: {df_ModelSelection_Breast.loc[ModelID-1,'Num Parameters']}")
    print(f"MLE: {df_ModelSelection_Breast.loc[ModelID-1,'MLE Parameter'][0]}")
    print(f"Likelihood value: {df_ModelSelection_Breast.loc[ModelID-1,'Max Likelihood']}  RSS: {df_ModelSelection_Breast.loc[ModelID-1,'RSS']}")
    print(f"AIC: {df_ModelSelection_Breast.loc[ModelID-1,'weight AIC']}  AIC_c: {df_ModelSelection_Breast.loc[ModelID-1,'weight AICc']} BIC: {df_ModelSelection_Breast.loc[ModelID-1,'weight BIC']}")

def Plot_MLE_Solution_Mets(ModelID):
    Plot_curves_mets(ModelID)
    print(f"Number of parameter: {df_ModelSelection_Mets.loc[ModelID-1,'Num Parameters']}")
    print(f"MLE: {df_ModelSelection_Mets.loc[ModelID-1,'MLE Parameter']}")
    print(f"Likelihood value: {df_ModelSelection_Mets.loc[ModelID-1,'Max Likelihood']}  RSS: {df_ModelSelection_Mets.loc[ModelID-1,'RSS']}")
    print(f"AIC: {df_ModelSelection_Mets.loc[ModelID-1,'weight AIC']}  AIC_c: {df_ModelSelection_Mets.loc[ModelID-1,'weight AICc']} BIC: {df_ModelSelection_Mets.loc[ModelID-1,'weight BIC']}")

def Plot_Chains(ModelID, idx, labels, Samples_Chains, Samples_Chains_loglik, parameters_df):
    samples = Samples_Chains[ModelID-1]
    samples_loglik = Samples_Chains_loglik[ModelID-1]
    new_lowerBounds = parameters_df["Lower bound"].loc[idx].tolist()
    new_upperBounds = parameters_df["Upper bound"].loc[idx].tolist()
    fig, axes = plt.subplots(len(idx),2, gridspec_kw={'width_ratios': [3, 1]}, figsize=(10, 14))
    for i in range(len(idx)): # loop parameters
         # create the corresponding number of labels (= the text you want to display)
        ax1 = axes[i,0]
        ax2 = axes[i,1]
        for j in range(samples.shape[1]):
            ax1.plot(samples[:, j, i], alpha=0.3)
            sns.kdeplot(samples[:, j, i], ax=ax2)
        if (i != len(idx)-1): ax1.axes.xaxis.set_ticklabels([])
        ax1.axvline(x=MCMC_Burn_in, color='r') # Burn in line
        ax1.set_xlim(0, len(samples))
        ax1.set_ylim(new_lowerBounds[i], new_upperBounds[i])
        ax1.set_ylabel(labels[i])
        ax1.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1,0].set_xlabel("step number");

def Plot_Posterior(ModelID, labels, FlatSamples):
    flat_samples = FlatSamples[ModelID-1]
    df = pd.DataFrame(flat_samples, columns=labels)
    df["chain"] = 0
    df["draw"] = np.arange(flat_samples.shape[0], dtype=int)
    df = df.set_index(["chain", "draw"])
    xdata = xr.Dataset.from_dataframe(df)
    dataset = az.convert_to_inference_data(xdata)
    az.plot_posterior(dataset,hdi_prob=.90,point_estimate='mode')

def PlotResultBayes_breast(ModelID, Plot):
    widgets.Output().clear_output()
    idx = Apply_Constraints(constraints_breast[ModelID-1], parameters_breast_df)
    labels = parameters_breast_df["Parameter"].iloc[idx].tolist()
    if (Plot == 'Solution'): Plot_MLE_Solution_Breast(ModelID)
    if (Plot == 'MCMC Chains'): Plot_Chains(ModelID, idx, labels, Samples_Chains_breast, Samples_Chains_loglik_breast, parameters_breast_df)
    if (Plot == 'Posterior dist'): Plot_Posterior(ModelID, labels, FlatSamples_breast)
    plt.show()

def PlotResultBayes_mets(ModelID, Plot):
    widgets.Output().clear_output()
    idx = Apply_Constraints(constraints_mets[ModelID-1], parameters_mets_df)
    labels = parameters_mets_df["Parameter"].iloc[idx].tolist()
    if (Plot == 'Solution'): Plot_MLE_Solution_Mets(ModelID)
    if (Plot == 'MCMC Chains'): Plot_Chains(ModelID, idx, labels, Samples_Chains_mets, Samples_Chains_loglik_mets, parameters_mets_df)
    if (Plot == 'Posterior dist'): Plot_Posterior(ModelID, labels, FlatSamples_mets)
    plt.show()

def PlotSelection(df_ModelSelection):
#     widgets.Output().clear_output()
    def PlotSelec(dataframe,labelSimilarity):
        fig, axs = plt.subplots(1, 2,figsize=(18, 6))
        axs[0].stem(dataframe["ModelID"], dataframe["weight AIC"], label='AIC', linefmt="C0-", basefmt=" ", markerfmt="C0o", use_line_collection=True)
        axs[0].stem(dataframe["ModelID"], dataframe["weight AICc"], label='AICc', linefmt="C3-", basefmt=" ", markerfmt="C3o", use_line_collection=True)
        axs[0].stem(dataframe["ModelID"], dataframe["weight BIC"], label='BIC', linefmt="C2-", basefmt=" ", markerfmt="C2o", use_line_collection=True)
#         axs[0].stem(dataframe["ModelID"], dataframe["weight AICc_RSS"], label='AIC_RSS', linefmt="C1-", basefmt=" ", markerfmt="C1o", use_line_collection=True)
        axs[0].set(xlabel = "Model ID", ylabel = "Weight from Information Criterion", title="Information Criterion")
        axs[0].legend()
        # Complexity vs Accuracy
        idx_best_AIC = dataframe["weight AIC"].idxmax()
        idx_best_AIC_c = dataframe["weight AICc"].idxmax()
        idx_best_BIC = dataframe["weight BIC"].idxmax()
        idx_other = dataframe.index.tolist()
        idx_other.remove(idx_best_AIC)
        if (idx_best_AIC_c in idx_other): idx_other.remove(idx_best_AIC_c)
        if (idx_best_BIC in idx_other): idx_other.remove(idx_best_BIC)
        axs[1].plot(dataframe['Num Parameters'].iloc[idx_other], dataframe[labelSimilarity].iloc[idx_other], "ko", alpha = 0.2, markersize = 10, label = "Other model")
        axs[1].plot(dataframe['Num Parameters'].iloc[idx_best_AIC], dataframe[labelSimilarity].iloc[idx_best_AIC], "C0*", markersize = 15, label = "Best model AIC: "+str(dataframe['ModelID'].iloc[idx_best_AIC]))
        axs[1].plot(dataframe['Num Parameters'].iloc[idx_best_AIC_c], dataframe[labelSimilarity].iloc[idx_best_AIC_c], "C3*", alpha = 0.5, markersize = 15, label = "Best model AICc: "+str(dataframe['ModelID'].iloc[idx_best_AIC_c]))
        axs[1].plot(dataframe['Num Parameters'].iloc[idx_best_BIC], dataframe[labelSimilarity].iloc[idx_best_BIC], "C2*", alpha = 0.5, markersize = 15, label = "Best model BIC: "+str(dataframe['ModelID'].iloc[idx_best_BIC]))
        axs[1].ticklabel_format(axis='y', style='sci', scilimits=(-1,1), useMathText=True)
        axs[1].set(xlabel = "Model complexity", ylabel = "Maximum likelihood value", title="Complexity vs Accuracy")
        axs[1].legend()
        #plt.subplots_adjust(hspace=0.4)
        print(f"Model {dataframe.loc[dataframe['RSS'].idxmin(),'ModelID']} has the lower RSS!")
        print(f"Model {dataframe.loc[dataframe[labelSimilarity].idxmax(),'ModelID']} has the higher {labelSimilarity}!")
        plt.show()
    PlotSelec(dataframe=df_ModelSelection, labelSimilarity='Max Likelihood')
    df = df_ModelSelection.sort_values(by=['weight AICc'],ascending=False,ignore_index=True)
    display(df[['ModelID','Num Parameters','Max Likelihood','RSS','Delta AICc','weight AICc', 'Evidence ratio AICc']])

def Evidence_Constraints_breast():
    ndi = 'tau = 0'; eg = 'K_b = inf'
    h1 = 'alpha_bg = alpha_br'; h2 = 'beta_g = beta_r'
    all_IDs = range(1,len(constraints_breast)+1)
    eg_IDs = [i+1 for i,x in enumerate(constraints_breast) if eg in x]
    lg_IDs = list(set(all_IDs) - set(eg_IDs))
    ndi_IDs = [i+1 for i,x in enumerate(constraints_breast) if ndi in x]
    di_IDs = list(set(all_IDs) - set(ndi_IDs))
    h1_IDs = [i+1 for i,x in enumerate(constraints_breast) if h1 in x]
    h2_IDs = [i+1 for i,x in enumerate(constraints_breast) if h2 in x]
    return {'EG': eg_IDs, 'LG':lg_IDs, 'NDI':ndi_IDs,'DI':di_IDs, 'H1':h1_IDs, 'H2':h2_IDs}

def Evidence_Constraints_mets():
    eg = 'K_l = inf'
    h3 = 'phi_r = 3*phi_g'; h4 = 'f_Eg = f_Er'; h5 = 'alpha_lg = alpha_lr'
    all_IDs = range(1,len(constraints_breast)+1)
    eg_IDs = [i+1 for i,x in enumerate(constraints_mets) if eg in x]
    lg_IDs = list(set(all_IDs) - set(eg_IDs))
    h3_IDs = [i+1 for i,x in enumerate(constraints_mets) if h3 in x]
    h4_IDs = [i+1 for i,x in enumerate(constraints_mets) if h4 in x]
    h5_IDs = [i+1 for i,x in enumerate(constraints_mets) if h5 in x]
    return {'EG': eg_IDs, 'LG':lg_IDs, 'H3':h3_IDs, 'H4':h4_IDs, 'H5':h5_IDs}



def GUI_ModeSelectionData():
    import ipywidgets as widgets
    from ipywidgets import GridspecLayout
    from IPython.display import display, Math

    # Randy
    homedir = os.getcwd()
    nanoHUB_flag = False
    if( 'HOME' in os.environ.keys() ):
        nanoHUB_flag = "home/nanohub" in os.environ['HOME']

    # define a Layout giving 50px margin between the items.
    item_layout = widgets.Layout(margin='0 0 50px 0')
    # define accordion
    Breast_output = widgets.Output(); Mets_output = widgets.Output()
    # internal tabs
    Obs_data_Output = widgets.Output(); Comp_model_Output = widgets.Output(); Parameters_Output = widgets.Output(); Constraints_Output = widgets.Output(); ModelSelection_Output = widgets.Output(); Calibration_Output = widgets.Output()
    Obs_data_Output_mets = widgets.Output(); Comp_model_Output_mets = widgets.Output(); Parameters_Output_mets = widgets.Output(); Constraints_Output_mets = widgets.Output(); ModelSelection_Output_mets = widgets.Output(); Calibration_Output_mets = widgets.Output()
    # define slider
    SliderModel = widgets.IntSlider(min=1, max=len(constraints_breast), step=1, value=1, description="Model")
    FloatTextResult = widgets.FloatText()
    SliderModel_mets = widgets.IntSlider(min=1, max=len(constraints_mets), step=1, value=1, description="Model")
    FloatTextResult_mets = widgets.FloatText()
    # link between FloatText and slider
    mylink = widgets.jslink((SliderModel, 'value'), (FloatTextResult, 'value'))
    mylink_mets = widgets.jslink((SliderModel_mets, 'value'), (FloatTextResult_mets, 'value'))
    # dropbox in calibration
    DropDownPlot = widgets.Dropdown( options=['Solution', 'MCMC Chains', 'Posterior dist'], value='Solution', description='Plot:', disabled=False)
    DropDownPlot_mets = widgets.Dropdown( options=['Solution', 'MCMC Chains', 'Posterior dist'], value='Solution', description='Plot:', disabled=False)
    def common_filtering():
        with Obs_data_Output:
            Plot_curves_breast()
        with Obs_data_Output_mets:
            Plot_curves_mets()
        with Comp_model_Output:
            ModelsStructures_breast()
        with Comp_model_Output_mets:
            ModelsStructures_mets()
        with Parameters_Output:
            display(parameters_breast_df)
        with Parameters_Output_mets:
            display(parameters_mets_df)
        with Constraints_Output:
            display(Grid_Constraints(constraints_breast))
        with Constraints_Output_mets:
            display(Grid_Constraints(constraints_mets))
        with ModelSelection_Output:
            PlotSelection(df_ModelSelection_Breast)
            dic_evidence = Evidence_Constraints_breast()
            dic_weight = {}
            for key, value in dic_evidence.items():
                df_evidence = df_ModelSelection_Breast.loc[ df_ModelSelection_Breast['ModelID'].isin(value)]
                dic_weight[key] = df_evidence[['weight AICc']].sum(axis=0)[0]
            print("Relative hypotheses importance")
            display(pd.DataFrame(dic_weight, index=[0]))
        with ModelSelection_Output_mets:
            PlotSelection(df_ModelSelection_Mets)
            dic_evidence = Evidence_Constraints_mets()
            dic_weight = {}
            for key, value in dic_evidence.items():
                df_evidence = df_ModelSelection_Mets.loc[ df_ModelSelection_Mets['ModelID'].isin(value)]
                dic_weight[key] = df_evidence[['weight AICc']].sum(axis=0)[0]
            print("Relative hypotheses importance:")
            display(pd.DataFrame(dic_weight, index=[0]))
        with Calibration_Output:
            outPlotBayes = widgets.interactive_output(PlotResultBayes_breast, {'ModelID': SliderModel, 'Plot': DropDownPlot})
            display(widgets.HBox([SliderModel,FloatTextResult,DropDownPlot]))
            display(outPlotBayes)
            outTab = widgets.interactive_output(TabResult_breast, {'ModelID': SliderModel})
            display(outTab)
        with Calibration_Output_mets:
            outPlotBayes_mets = widgets.interactive_output(PlotResultBayes_mets, {'ModelID': SliderModel_mets, 'Plot': DropDownPlot_mets})
            display(widgets.HBox([SliderModel_mets,FloatTextResult_mets,DropDownPlot_mets]))
            display(outPlotBayes_mets)
            outTab_mets = widgets.interactive_output(TabResult_mets, {'ModelID': SliderModel_mets})
            display(outTab_mets)

    # create a container for the output with Tabs.
    tabs_breast = widgets.Tab([Obs_data_Output, Comp_model_Output, Parameters_Output, Constraints_Output, ModelSelection_Output, Calibration_Output],layout=item_layout)
    tabs_breast.set_title(0, 'Observational Data')
    tabs_breast.set_title(1, 'Model Candidates')
    tabs_breast.set_title(2, 'Parameters')
    tabs_breast.set_title(3, 'Constraints')
    tabs_breast.set_title(4, 'Model Selection')
    tabs_breast.set_title(5, 'Calibration')

    tabs_mets = widgets.Tab([Obs_data_Output_mets, Comp_model_Output_mets, Parameters_Output_mets, Constraints_Output_mets, ModelSelection_Output_mets, Calibration_Output_mets])
    tabs_mets.set_title(0, 'Observational Data')
    tabs_mets.set_title(1, 'Model Candidates')
    tabs_mets.set_title(2, 'Parameters')
    tabs_mets.set_title(3, 'Constraints')
    tabs_mets.set_title(4, 'Model Selection')
    tabs_mets.set_title(5, 'Calibration')

    accordion_ModelSelection = widgets.Accordion(children=[tabs_breast, tabs_mets])
    accordion_ModelSelection.set_title(0, 'Model Selection Breast')
    accordion_ModelSelection.set_title(1, 'Model Selection Metastasis')

    # stack the input widgets and the tab on top of each other with a VBox.
    dashboard = widgets.VBox([accordion_ModelSelection])
    common_filtering()
    display(dashboard)

if __name__ == '__main__':
    Plot_curves_breast(ModelID=1, FontSize = 20, FileName="Model01_Breast.svg")
    Plot_curves_breast(ModelID=4, FontSize = 20, FileName="Model04_Breast.svg")
    Plot_curves_breast(ModelID=8, FontSize = 20, FileName="Model08_Breast.svg")
    Plot_curves_breast(ModelID=14, FontSize = 20, FileName="Model14_Breast_Best.svg")

    Plot_curves_mets(ModelID=1, FontSize = 20, FileName="Model01_Mets.svg")
    Plot_curves_mets(ModelID=9, FontSize = 20, FileName="Model09_Mets.svg")
    Plot_curves_mets(ModelID=11, FontSize = 20, FileName="Model11_Mets_Best.svg")
    Plot_curves_mets(ModelID=13, FontSize = 20, FileName="Model13_Mets.svg")

from matplotlib.ticker import ScalarFormatter
import seaborn as sns
import arviz as az
import xarray as xr
from MCMC import *
from ModelSelection import *
import pickle
import os

# Result MCMC Breast
MCMC_Burn_in = 100
folder = "data/Model_Candidates/"
# Breast
NumPar_breast, MLE_Par_breast, MLE_Solution_breast , Median_Par_breast, Mean_Par_breast, LL_value_breast, RSS_value_breast, FlatSamples_breast, Samples_Chains_breast, Samples_Chains_loglik_breast = Loading_MCMC_breast(folder, burn_in = MCMC_Burn_in)
dataSize_breast = len(obs_data.DsRedBreastMean)+len(obs_data.GFPBreastMean)
AIC_breast, AICc_breast, BIC_breast =  Model_Selection(MLE_Par_breast, LL_value_breast, dataSize_breast, len(constraints_breast)) # Using log likelihood
w_AIC_breast, Delta_AIC_breast = weight_InfCriterion(AIC_breast)
ER_AIC_breast = evidence_ratio(Delta_AIC_breast)
w_AICc_breast, Delta_AICc_breast = weight_InfCriterion(AICc_breast)
ER_AICc_breast = evidence_ratio(Delta_AICc_breast)
w_BIC_breast, Delta_BIC_breast = weight_InfCriterion(BIC_breast)
ER_BIC_breast = evidence_ratio(Delta_BIC_breast)
df_ModelSelection_Breast = pd.DataFrame(data={'ModelID': range(1,len(constraints_breast)+1), 'Num Parameters': NumPar_breast, 'MLE Parameter': MLE_Par_breast, 'Mean Parameter': Mean_Par_breast, 'Median Parameter': Median_Par_breast, 'MLE Smooth Solution': MLE_Solution_breast, 'Max Likelihood': np.exp(LL_value_breast), 'RSS': RSS_value_breast, 'Delta AIC': Delta_AIC_breast, 'weight AIC': w_AIC_breast, 'Evidence ratio AIC': ER_AIC_breast, 'Delta AICc': Delta_AICc_breast, 'weight AICc': w_AICc_breast, 'Evidence ratio AICc': ER_AICc_breast, 'Delta BIC': Delta_BIC_breast, 'weight BIC': w_BIC_breast, 'Evidence ratio BIC': ER_BIC_breast, 'Hypotheses': Constraints_to_hypothesis(constraints_breast)} )
# Bloodstream and Lung
Model_Parameters_Breast = {18: df_ModelSelection_Breast.loc[18-1,'MLE Parameter'][0], 22: df_ModelSelection_Breast.loc[22-1,'MLE Parameter'][0]}
NumPar_mets=[]; MLE_Par_mets=[]; MLE_Solution_mets =[]; Median_Par_mets=[]; Mean_Par_mets=[]; LL_value_mets=[]; RSS_value_mets=[]; FlatSamples_mets=[]; Samples_Chains_mets=[]; Samples_Chains_loglik_mets = []; ModelID_B = []
for modelID_breast, theta_breast in Model_Parameters_Breast.items():
    NumPar_mets_curr, MLE_Par_mets_curr, MLE_Solution_mets_curr , Median_Par_mets_curr, Mean_Par_mets_curr, LL_value_mets_curr, RSS_value_mets_curr, FlatSamples_mets_curr, Samples_Chains_mets_curr, Samples_Chains_loglik_mets_curr = Loading_MCMC_mets(folder,theta_breast, modelID_breast, burn_in = MCMC_Burn_in)
    NumPar_mets += NumPar_mets_curr; MLE_Par_mets += MLE_Par_mets_curr; MLE_Solution_mets += MLE_Solution_mets_curr; Median_Par_mets += Median_Par_mets_curr; Mean_Par_mets += Mean_Par_mets_curr; LL_value_mets += LL_value_mets_curr; RSS_value_mets += RSS_value_mets_curr; FlatSamples_mets += FlatSamples_mets_curr; Samples_Chains_mets += Samples_Chains_mets_curr; Samples_Chains_loglik_mets += Samples_Chains_loglik_mets_curr; ModelID_B += len(NumPar_mets_curr)*[modelID_breast]
dataSize_mets = len(obs_data.DsRedBloodStMean)+len(obs_data.GFPBloodStMean)+len(obs_data.DsRedLungMean)+len(obs_data.GFPLungMean)
AIC_mets, AICc_mets, BIC_mets =  Model_Selection(MLE_Par_mets, LL_value_mets, dataSize_mets, len(ModelID_B)) # Using log likelihood
w_AIC_mets, Delta_AIC_mets = weight_InfCriterion(AIC_mets)
ER_AIC_mets = evidence_ratio(Delta_AIC_mets)
w_AICc_mets, Delta_AICc_mets = weight_InfCriterion(AICc_mets)
ER_AICc_mets = evidence_ratio(Delta_AICc_mets)
w_BIC_mets, Delta_BIC_mets = weight_InfCriterion(BIC_mets)
ER_BIC_mets = evidence_ratio(Delta_BIC_mets)
df_ModelSelection_Mets = pd.DataFrame(data={'ModelID': range(1,len(ModelID_B)+1),'ModelID breast': ModelID_B, 'Num Parameters': NumPar_mets, 'MLE Parameter': MLE_Par_mets, 'Mean Parameter': Mean_Par_mets, 'Median Parameter': Median_Par_mets, 'MLE Smooth Solution': MLE_Solution_mets, 'Max Likelihood': np.exp(LL_value_mets), 'RSS': RSS_value_mets, 'Delta AIC': Delta_AIC_mets, 'weight AIC': w_AIC_mets, 'Evidence ratio AIC': ER_AIC_mets, 'Delta AICc': Delta_AICc_mets, 'weight AICc': w_AICc_mets, 'Evidence ratio AICc': ER_AICc_mets, 'Delta BIC': Delta_BIC_mets, 'weight BIC': w_BIC_mets, 'Evidence ratio BIC': ER_BIC_mets, 'Hypotheses': len(Model_Parameters_Breast)*Constraints_to_hypothesis(constraints_mets)} )
# Results from microscale
dic_calib_PhysiCell = {}
dic_cluters_PhysiCell = {}
dic_images3D_PhysiCell = {}
for modelID in [68,89]:
    for hypoID in [1,2]:
        Calib_PhysiCell = 'data/Calib_PhysiCell_M%02d_H%d.pkl'%(modelID, hypoID)
        Clusters_PhysiCell = 'data/Clusters_M%02d_H%d.pkl'%(modelID, hypoID)
        Images3D_PhysiCell = 'data/3Dimages_M%02d_H%d.pkl'%(modelID, hypoID)
        with open(Calib_PhysiCell, 'rb') as file:
            dic_calib_PhysiCell['M%02d_H%d'%(modelID, hypoID)] = pickle.load(file) # Pickle from pandas
        with open(Clusters_PhysiCell, 'rb') as file:
            dic_cluters_PhysiCell['M%02d_H%d'%(modelID, hypoID)] = pickle.load(file) # Pickle from pandas
        with open(Images3D_PhysiCell, 'rb') as file:
            dic_images3D_PhysiCell['M%02d_H%d'%(modelID, hypoID)] = pickle.load(file)

pd.set_option('display.max_rows', 100) # max rows to display
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

def Plot_curves_mets(ModelID=None, FontSize=None, FileName=None, PlotBreast=True):
    if PlotBreast: fig, (ax1,ax2,ax3) = plt.subplots(1, 3,figsize=(18, 4))
    else: fig, (ax2,ax3) = plt.subplots(1, 2,figsize=(12, 4))
    if (ModelID):
        output = df_ModelSelection_Mets.loc[ModelID-1,'MLE Smooth Solution']
        if PlotBreast:
            ax1.plot(smooth_time,output[:,0],color="red")
            ax1.plot(smooth_time,output[:,1],color="green")
        ax2.plot(smooth_time,output[:,2],color="red")
        ax2.plot(smooth_time,output[:,3],color="green")
        ax3.plot(smooth_time,output[:,4],color="red")
        ax3.plot(smooth_time,output[:,5],color="green")
    if PlotBreast:
        ax1.scatter(obs_data.obs_time,obs_data.DsRedBreastMean,color="red")
        ax1.scatter(obs_data.obs_time,obs_data.GFPBreastMean,color="green")
        ax1.errorbar(obs_data.obs_time[1:],obs_data.DsRedBreastMean[1:],yerr=obs_data.DsRedBreastStd[1:],ls='none',ecolor='red',capsize=10)
        ax1.errorbar(obs_data.obs_time[1:],obs_data.GFPBreastMean[1:],yerr=obs_data.GFPBreastStd[1:],ls='none',ecolor='green',capsize=10)
        # ax1.ticklabel_format(axis='y', style='sci', scilimits=(-1,1), useMathText=True)
        ax1.set(xlabel = "Time (days)", ylabel = "Number of cells", title = "Breast tumor")
    ax2.scatter([obs_data.obs_time[0], obs_data.obs_time[-1]], obs_data.DsRedBloodStMean, color='red')
    ax2.scatter([obs_data.obs_time[0], obs_data.obs_time[-1]], obs_data.GFPBloodStMean, color='green')
    ax2.errorbar(obs_data.obs_time[-1],obs_data.DsRedBloodStMean[-1], yerr=obs_data.DsRedBloodStStd[-1], ls='none',ecolor='red',capsize=10)
    ax2.errorbar(obs_data.obs_time[-1], obs_data.GFPBloodStMean[-1], yerr=obs_data.GFPBloodStStd[-1], ls='none',ecolor='green',capsize=10)
    # ax2.ticklabel_format(axis='y', style='sci', scilimits=(-1,1), useMathText=True)
    ax2.set(xlabel = "Time (days)", ylabel = "Number of cells", title = "Bloodstream")
    ax3.scatter(obs_data.obs_time,obs_data.DsRedLungMean,color="red")
    ax3.scatter(obs_data.obs_time,obs_data.GFPLungMean,color="green")
    ax3.errorbar(obs_data.obs_time[1:],obs_data.DsRedLungMean[1:],yerr=obs_data.DsRedLungStd[1:],ls='none',ecolor='red',capsize=10)
    ax3.errorbar(obs_data.obs_time[1:],obs_data.GFPLungMean[1:],yerr=obs_data.GFPLungStd[1:],ls='none',ecolor='green',capsize=10)
    # ax3.ticklabel_format(axis='y', style='sci', scilimits=(-1,1), useMathText=True)
    ax3.set(xlabel = "Time (days)", ylabel = "Number of cells", title = "Lung")#, yscale='symlog')
    if PlotBreast: axes = [ax1,ax2,ax3]
    else: axes = [ax2,ax3]
    for ax in axes:
        yScalarFormatter = ScalarFormatterClass(useMathText=True)
        yScalarFormatter.set_powerlimits((-1,1))
        ax.yaxis.set_major_formatter(yScalarFormatter)
        if FontSize:
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label, ax.yaxis.offsetText] + ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(FontSize)
    fig.tight_layout()
    if FileName: plt.savefig(FileName)
    else: plt.show()

def ModelsStructures_breast():
    import ipywidgets as widgets
    from IPython.display import display
    caption_size = 'h4'
    text1 = widgets.HTMLMath( value=r"<{size}>Let $[B_r]$ and $[B_g]$ the number of red and green cells in breast. The variation of cell population is given by:</{size}>".format(size=caption_size))
    text2 = widgets.HTMLMath( value=r"$$\begin{align} \frac{d[B_r]}{dt} &=& \mathcal{G}_{[B_r]} - \lambda [B_r] - \mathcal{I}_{[B_r]}\\ \frac{d[B_g]}{dt} &=& \mathcal{G}_{[B_g]} + \lambda [B_r] - \mathcal{I}_{[B_g]}\end{align}$$")
    title3 = widgets.HTML(value='<{size}> Subject to mathematical assumptions: </{size}>'.format(size=caption_size))
    text3 = widgets.HTMLMath( value=r"$$\begin{align} \fbox{EG}: \mathcal{G}_{[i]} &=& \alpha_{[i]}[i] \quad or \quad \fbox{LG}: \mathcal{G}_{[i]}=\alpha_{[i]}[i]\left( 1 - \frac{[i]}{K_{[i]}} \right), \quad i=\{B_r,B_g\}\\ \fbox{NDI}: \mathcal{I}_{[i]} &=& \beta_{[i]}[i] \quad or \quad \fbox{DI}: \mathcal{I}_{[i]}=\beta_{[i]} \mathcal{H}(t-\tau)[i],  \quad i=\{B_r,B_g \} \end{align}$$")
    text4 = widgets.HTMLMath( value=r"where $\fbox{EG}$ and $\fbox{LG}$ are exponential and logistic growth, $\fbox{DI}$ and $\fbox{NDI}$ are delay and no delay intravasation, respectively, and $\mathcal{H}(x) = \begin{cases} 1 & \text{if } x\geq0\\ 0 & \text{else} \end{cases}$ is the Heaviside function.")
    title5 = widgets.HTML(value='<{size}>Subject to parameters assumptions: </{size}>'.format(size=caption_size))
    text5 = widgets.HTMLMath( value=r"$$\begin{array}{llll} \fbox{H1}&:& \alpha_{[B_r]} = \alpha_{[B_g]} & \mbox{The growth rates of red and green cells are equal in breast tissue.}\\ \fbox{H2a}&:& \beta_{[B_r]} = \beta_{[B_g]} & \mbox{The intravasation rates of red and green cells are equal in breast tissue.}\\ \fbox{H2b}&:& \beta_{[B_g]} = 1.5*\beta_{[B_r]} & \mbox{The intravasation rate of red is one and half more than green cells in breast tissue.} \end{array}$$")
    display(text1,text2,title3,text3,text4,title5,text5)

def ModelsStructures_mets():
    import ipywidgets as widgets
    from IPython.display import display
    caption_size = 'h4'
    text1 = widgets.HTMLMath( value=r"<{size}>Let $[B_r]$, $[B_g]$, $[C_r]$, $[C_g]$, $[L_r]$, and $[L_g]$ the number of red and green cells in breast tumor, bloodstream, and lung. The variation of cell population is given by:</{size}>".format(size=caption_size))
    text2 = widgets.HTMLMath( value=r"$$\begin{align} \color{gray}{\frac{d[B_r]}{dt}} &\color{gray}{= \mathcal{G}_{[B_r]} - \lambda [B_r] - \mathcal{I}_{[B_r]}}\\ \color{gray}{\frac{d[B_g]}{dt}} &\color{gray}{= \mathcal{G}_{[B_g]} + \lambda [B_r] - \mathcal{I}_{[B_g]}}\\ \frac{d[C_r]}{dt} &= \mathcal{I}_{[B_r]} - \phi_{r}[C_r]\\ \frac{d[C_g]}{dt} &= \mathcal{I}_{[B_g]} - \phi_{g}[C_g]\\ \frac{d[L_r]}{dt} &= f_r \phi_{r}[C_r] + \mathcal{G}_{[L_r]}\\ \frac{d[L_g]}{dt} &= f_g \phi_{g}[C_g] + \mathcal{G}_{[L_g]} \end{align}$$")
    title3 = widgets.HTML(value='<{size}>Subject to mathematical assumptions: </{size}>'.format(size=caption_size))
    text3 = widgets.HTMLMath( value=r"$$\begin{align} \fbox{EG}: \mathcal{G}_{[i]} &=& \alpha_{[i]}[i] \quad or \quad \fbox{LG}: \mathcal{G}_{[i]}=\alpha_{[i]}[i]\left( 1 - \frac{[i]}{K_{[i]}} \right), \quad i=\{L_r,L_g\} \end{align}$$")
    text4 = widgets.HTMLMath( value=r"where $\fbox{EG}$ and $\fbox{LG}$ are exponential and logistic growth, respectively.")
    title5 = widgets.HTML(value='<{size}> Subject to parameters assumptions: </{size}>'.format(size=caption_size))
    text5 = widgets.HTMLMath( value=r"$$\begin{array}{llll} \fbox{H3a}&:& \phi_{r} = \phi_{g} & \mbox{The green and red circulating tumor cells (CTCs) intravasate or die equally.}\\ \fbox{H3b}&:& \phi_{r} = 3*\phi_{g} & \mbox{The green circulating tumor cells (CTCs) intravasate or die three times more than red CTCs.}\\ \fbox{H3c}&:& \phi_{r} = 6*\phi_{g} & \mbox{The green circulating tumor cells (CTCs) intravasate or die six times more than red CTCs.}\\ \fbox{H4a}&:& f_g\phi_{g} = f_r\phi_{r} & \mbox{The extravasation rates of red and green cells in the lung are equal.}\\ \fbox{H4b}&:& f_g\phi_{g} = 1.5*f_r\phi_{r} & \mbox{The extravasation rate of green cells into the lung is one and a half times greater than red cells.}\\ \fbox{H5}&:& \alpha_{[L_r]} = \alpha_{[L_g]} & \mbox{The growth rates of red and green cells are equal in the lung.} \end{array}$$")
    display(text1,text2,title3,text3,text4,title5,text5)

def data_frame_parameter_result(ID_model, param_value, constraints, parameters_df):
    idx = Apply_Constraints(constraints[ID_model-1],parameters_df)
    return pd.DataFrame(data={'Parameters': parameters_df["Parameter"].iloc[idx], 'Value': param_value, 'Unit': parameters_df["Unit"].iloc[idx], 'Description': parameters_df["Description"].iloc[idx]})

def setup_ui(df, height="auto", width="auto"):
    import ipywidgets as widgets
    from IPython.display import display
    out = widgets.Output(layout=widgets.Layout(height=height, width=width))
    with out:
        display(df)
    return out

def TabResult_breast(ModelID):
    import ipywidgets as widgets
    from IPython.display import display
    A = setup_ui(data_frame_parameter_result(ModelID, df_ModelSelection_Breast.loc[ModelID-1,'MLE Parameter'][0], constraints_breast, parameters_breast_df))
    value = '\n'.join(df_ModelSelection_Breast.loc[ModelID-1,'Hypotheses'])
    B = widgets.Textarea( value=value, description='Hypotheses:', disabled=True, layout=widgets.Layout(height="200px", width="auto"))
    display(widgets.HBox([A,B]))

def TabResult_mets(ModelID):
    import ipywidgets as widgets
    from IPython.display import display
    if ( len(constraints_mets) < ModelID ): ModelID_constraint = ModelID%(len(constraints_mets)+1) + 1
    else: ModelID_constraint = ModelID
    A = setup_ui(data_frame_parameter_result(ModelID_constraint, df_ModelSelection_Mets.loc[ModelID-1,'MLE Parameter'][0], constraints_mets, parameters_mets_df))
    value = '\n'.join(df_ModelSelection_Mets.loc[ModelID-1,'Hypotheses'])
    B = widgets.Textarea( value=value, description='Hypotheses:', disabled=True, layout=widgets.Layout(height="200px", width="auto"))
    display(widgets.HBox([A,B]))

def Plot_MLE_Solution_Breast(ModelID):
    Plot_curves_breast(ModelID)
    print(f"Number of parameter: {df_ModelSelection_Breast.loc[ModelID-1,'Num Parameters']}")
    print(f"Likelihood value: {df_ModelSelection_Breast.loc[ModelID-1,'Max Likelihood']}  RSS: {df_ModelSelection_Breast.loc[ModelID-1,'RSS']}")
    # print(f"AIC: {df_ModelSelection_Breast.loc[ModelID-1,'weight AIC']}  AIC_c: {df_ModelSelection_Breast.loc[ModelID-1,'weight AICc']} BIC: {df_ModelSelection_Breast.loc[ModelID-1,'weight BIC']}")

def Plot_MLE_Solution_Mets(ModelID):
    Plot_curves_mets(ModelID,PlotBreast=False)
    print(f"Number of parameter: {df_ModelSelection_Mets.loc[ModelID-1,'Num Parameters']}")
    print(f"Likelihood value: {df_ModelSelection_Mets.loc[ModelID-1,'Max Likelihood']}  RSS: {df_ModelSelection_Mets.loc[ModelID-1,'RSS']}")
    # print(f"AIC: {df_ModelSelection_Mets.loc[ModelID-1,'weight AIC']}  AIC_c: {df_ModelSelection_Mets.loc[ModelID-1,'weight AICc']} BIC: {df_ModelSelection_Mets.loc[ModelID-1,'weight BIC']}")

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
    idx = Apply_Constraints(constraints_breast[ModelID-1], parameters_breast_df)
    labels = parameters_breast_df["Parameter"].iloc[idx].tolist()
    if (Plot == 'Solution'): Plot_MLE_Solution_Breast(ModelID)
    if (Plot == 'MCMC Chains'): Plot_Chains(ModelID, idx, labels, Samples_Chains_breast, Samples_Chains_loglik_breast, parameters_breast_df)
    if (Plot == 'Posterior dist'): Plot_Posterior(ModelID, labels, FlatSamples_breast)
    plt.show()

def PlotResultBayes_mets(ModelID, Plot):
    if ( len(constraints_mets) < ModelID ): ModelID_constraint = ModelID%(len(constraints_mets)+1) + 1
    else: ModelID_constraint = ModelID
    idx = Apply_Constraints(constraints_mets[ModelID_constraint-1], parameters_mets_df)
    labels = parameters_mets_df["Parameter"].iloc[idx].tolist()
    if (Plot == 'Solution'): Plot_MLE_Solution_Mets(ModelID)
    if (Plot == 'MCMC Chains'): Plot_Chains(ModelID, idx, labels, Samples_Chains_mets, Samples_Chains_loglik_mets, parameters_mets_df)
    if (Plot == 'Posterior dist'): Plot_Posterior(ModelID, labels, FlatSamples_mets)
    plt.show()

def PlotSelection(dataframe, labelSimilarity='Max Likelihood', FontSize=None, FileName=None, title = True):
    fig, axs = plt.subplots(1, 2,figsize=(12, 4))
    markerline1, stemline1, baseline1, = axs[0].stem(dataframe["ModelID"], dataframe["weight AIC"], label='AIC', linefmt="C0-", basefmt=" ", markerfmt="C0o", use_line_collection=True)
    markerline2, stemline2, baseline2, = axs[0].stem(dataframe["ModelID"], dataframe["weight AICc"], label='AICc', linefmt="C3-", basefmt=" ", markerfmt="C3o", use_line_collection=True)
    markerline3, stemline3, baseline3, = axs[0].stem(dataframe["ModelID"], dataframe["weight BIC"], label='BIC', linefmt="C2-", basefmt=" ", markerfmt="C2o", use_line_collection=True)
    # plt.setp(markerline1,markersize = 4.0)
    # plt.setp(markerline2,markersize = 3.0)
    # plt.setp(markerline3,markersize = 2.5)
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
    axs[1].plot(dataframe['Num Parameters'].iloc[idx_best_AIC], dataframe[labelSimilarity].iloc[idx_best_AIC], "C0s", markersize = 10, label = "Best model AIC: "+str(dataframe['ModelID'].iloc[idx_best_AIC]))
    axs[1].plot(dataframe['Num Parameters'].iloc[idx_best_AIC_c], dataframe[labelSimilarity].iloc[idx_best_AIC_c], "C3p", alpha = 1.0, markersize = 10, label = "Best model AICc: "+str(dataframe['ModelID'].iloc[idx_best_AIC_c]))
    axs[1].plot(dataframe['Num Parameters'].iloc[idx_best_BIC], dataframe[labelSimilarity].iloc[idx_best_BIC], "C2*", alpha = 1.0, markersize = 10, label = "Best model BIC: "+str(dataframe['ModelID'].iloc[idx_best_BIC]))
    axs[1].ticklabel_format(axis='y', style='sci', scilimits=(-1,1), useMathText=True)
    axs[1].set_xticks=np.arange(dataframe['Num Parameters'].min(), dataframe['Num Parameters'].max()+1,step=1)
    axs[1].legend()
    #plt.subplots_adjust(hspace=0.4)
    if (title):
        axs[0].set(xlabel = "Model ID", ylabel = "Weight from Information Criterion", title="Information Criterion")
        axs[1].set(xlabel = "Model complexity", ylabel = "Maximum likelihood value", title="Complexity vs Accuracy")
    else:
        axs[0].set(xlabel = "Model ID", ylabel = r"$w_i[IC]$")
        axs[1].set(xlabel = "k", ylabel = r"$\pi_{like}(\bf{y}\mid\bf{\hat{\theta}})$")
    print(f"Model {dataframe.loc[dataframe['RSS'].idxmin(),'ModelID']} has the lower RSS!")
    print(f"Model {dataframe.loc[dataframe[labelSimilarity].idxmax(),'ModelID']} has the higher {labelSimilarity}!")
    if FontSize:
        for ax in axs:
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label, ax.yaxis.offsetText] + ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(FontSize)
    if FileName:
        fig.tight_layout()
        plt.savefig(FileName)
    else: plt.show()

def TabsSelection(df_ModelSelection, IC, Evidence_Function):
    from IPython.display import display
    import ipywidgets as widgets
    Delta = "Delta "+IC
    Weight = "weight "+IC
    df = df_ModelSelection.sort_values(by=[Weight],ascending=False,ignore_index=True)
    if ('ModelID breast' in df.columns): df_rank = df[['ModelID','ModelID breast','Num Parameters','Max Likelihood','RSS',Delta,Weight, 'Hypotheses']]
    else: df_rank = df[['ModelID','Num Parameters','Max Likelihood','RSS',Delta,Weight, 'Hypotheses']]
    # Plot relative importance of each hypothesis
    dic_evidence = Evidence_Function()
    dic_weight = {}
    for key, value in dic_evidence.items():
        df_evidence = df_ModelSelection.loc[ df_ModelSelection['ModelID'].isin(value)]
        dic_weight[key] = df_evidence[[Weight]].sum(axis=0)[0]
    display(widgets.VBox([ setup_ui(df_rank), widgets.HTML(value="Relative importance of each hypothesis:"), setup_ui(pd.DataFrame(dic_weight, index=[' '])) ]))

def Plot_Pie_Charts(df_ModelSelection, IC, Evidence_Function, Breast, FileName=None):
    Weight = "weight "+IC
    # Plot relative importance of each hypothesis
    dic_evidence = Evidence_Function()
    dic_weight = {}
    for key, value in dic_evidence.items():
        df_evidence = df_ModelSelection.loc[ df_ModelSelection['ModelID'].isin(value)]
        dic_weight[key] = df_evidence[[Weight]].sum(axis=0)[0]

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    grayscale = ['#4c4c4c','#848484','#dddddd','#bcbcbc']
    specs = [[{'type':'domain'}, {'type':'domain'}, {'type':'domain'}, {'type':'domain'}]]
    fig = make_subplots(rows=1, cols=4, specs=specs)
    labels_growth = ['EG','LG']
    values_growth = [dic_weight['EG'],dic_weight['LG']]
    fig.add_trace(go.Pie(labels=labels_growth, values=values_growth, textinfo='label+percent', insidetextorientation='radial',marker_colors=grayscale), 1, 1)
    if (Breast):
        labels_di = ['NDI','DI']
        values_di = [dic_weight['NDI'],dic_weight['DI']]
        labels_h1 = ['H1','None']
        values_h1 = [dic_weight['H1'],1-dic_weight['H1']]
        labels_h2 = ['H2a','H2b','None']
        values_h2 = [dic_weight['H2a'],dic_weight['H2b'],1-(dic_weight['H2a']+dic_weight['H2b'])]
        fig.add_trace(go.Pie(labels=labels_di, values=values_di, textinfo='label+percent', insidetextorientation='radial',marker_colors=grayscale), 1, 2)
        fig.add_trace(go.Pie(labels=labels_h1, values=values_h1, textinfo='label+percent', insidetextorientation='radial',marker_colors=grayscale), 1, 3)
        fig.add_trace(go.Pie(labels=labels_h2, values=values_h2, textinfo='label+percent', insidetextorientation='radial',marker_colors=grayscale), 1, 4)
        fig.update_layout(# Add annotations in the center of the donut pies.
            annotations=[dict(text='Growth', x=0.067, y=0.5, font_size=16, showarrow=False),
            dict(text='Delay', x=0.37, y=0.5, font_size=16, showarrow=False),
            dict(text='H1', x=0.63, y=0.5, font_size=16, showarrow=False),
            dict(text='H2', x=0.91, y=0.5, font_size=16, showarrow=False)],
            showlegend=False)
    else:
        labels_h3 = ['H3a','H3b','H3c','None']
        values_h3 = [dic_weight['H3a'],dic_weight['H3b'],dic_weight['H3c'],1-(dic_weight['H3a']+dic_weight['H3b']+dic_weight['H3c'])]
        labels_h4 = ['H4a','H4b','None']
        values_h4 = [dic_weight['H4a'],dic_weight['H4b'],1-(dic_weight['H4a']+dic_weight['H4b'])]
        labels_h5 = ['H5','None']
        values_h5 = [dic_weight['H5'],1-dic_weight['H5']]
        fig.add_trace(go.Pie(labels=labels_h3, values=values_h3, textinfo='label+percent', insidetextorientation='radial',marker_colors=grayscale), 1, 2)
        fig.add_trace(go.Pie(labels=labels_h4, values=values_h4, textinfo='label+percent', insidetextorientation='radial',marker_colors=grayscale), 1, 3)
        fig.add_trace(go.Pie(labels=labels_h5, values=values_h5, textinfo='label+percent', insidetextorientation='radial',marker_colors=grayscale), 1, 4)
        fig.update_layout(# Add annotations in the center of the donut pies.
            annotations=[dict(text='Growth', x=0.067, y=0.5, font_size=16, showarrow=False),
            dict(text='H3', x=0.37, y=0.5, font_size=16, showarrow=False),
            dict(text='H4', x=0.63, y=0.5, font_size=16, showarrow=False),
            dict(text='H5', x=0.91, y=0.5, font_size=16, showarrow=False)],
            showlegend=False)
    fig.update_traces(hole=.4) # Use `hole` to create a donut-like pie chart
    if FileName: fig.write_image(FileName)
    else: fig.show()

def Evidence_Constraints_breast(): # return model ID of each constraint
    eg_IDs = [ df_ModelSelection_Breast["ModelID"].loc[i] for i in range(len(df_ModelSelection_Breast.index)) if 'EG' in df_ModelSelection_Breast["Hypotheses"].loc[i] ]
    lg_IDs = [ df_ModelSelection_Breast["ModelID"].loc[i] for i in range(len(df_ModelSelection_Breast.index)) if 'LG' in df_ModelSelection_Breast["Hypotheses"].loc[i] ]
    ndi_IDs = [ df_ModelSelection_Breast["ModelID"].loc[i] for i in range(len(df_ModelSelection_Breast.index)) if 'NDI' in df_ModelSelection_Breast["Hypotheses"].loc[i] ]
    di_IDs =[ df_ModelSelection_Breast["ModelID"].loc[i] for i in range(len(df_ModelSelection_Breast.index)) if 'DI' in df_ModelSelection_Breast["Hypotheses"].loc[i] ]
    h1_IDs = [ df_ModelSelection_Breast["ModelID"].loc[i] for i in range(len(df_ModelSelection_Breast.index)) if 'H1' in df_ModelSelection_Breast["Hypotheses"].loc[i] ]
    h2a_IDs = [ df_ModelSelection_Breast["ModelID"].loc[i] for i in range(len(df_ModelSelection_Breast.index)) if 'H2a' in df_ModelSelection_Breast["Hypotheses"].loc[i] ]
    h2b_IDs = [ df_ModelSelection_Breast["ModelID"].loc[i] for i in range(len(df_ModelSelection_Breast.index)) if 'H2b' in df_ModelSelection_Breast["Hypotheses"].loc[i] ]
    return {'EG': eg_IDs, 'LG':lg_IDs, 'NDI':ndi_IDs,'DI':di_IDs, 'H1':h1_IDs, 'H2a':h2a_IDs, 'H2b':h2b_IDs}

def Evidence_Constraints_mets(): # return model ID of with each constraint
    eg_IDs = [ df_ModelSelection_Mets["ModelID"].loc[i] for i in range(len(df_ModelSelection_Mets.index)) if 'EG' in df_ModelSelection_Mets["Hypotheses"].loc[i] ]
    lg_IDs = [ df_ModelSelection_Mets["ModelID"].loc[i] for i in range(len(df_ModelSelection_Mets.index)) if 'LG' in df_ModelSelection_Mets["Hypotheses"].loc[i] ]
    h3a_IDs = [ df_ModelSelection_Mets["ModelID"].loc[i] for i in range(len(df_ModelSelection_Mets.index)) if 'H3a' in df_ModelSelection_Mets["Hypotheses"].loc[i] ]
    h3b_IDs = [ df_ModelSelection_Mets["ModelID"].loc[i] for i in range(len(df_ModelSelection_Mets.index)) if 'H3b' in df_ModelSelection_Mets["Hypotheses"].loc[i] ]
    h3c_IDs = [ df_ModelSelection_Mets["ModelID"].loc[i] for i in range(len(df_ModelSelection_Mets.index)) if 'H3c' in df_ModelSelection_Mets["Hypotheses"].loc[i] ]
    h4a_IDs = [ df_ModelSelection_Mets["ModelID"].loc[i] for i in range(len(df_ModelSelection_Mets.index)) if 'H4a' in df_ModelSelection_Mets["Hypotheses"].loc[i] ]
    h4b_IDs = [ df_ModelSelection_Mets["ModelID"].loc[i] for i in range(len(df_ModelSelection_Mets.index)) if 'H4b' in df_ModelSelection_Mets["Hypotheses"].loc[i] ]
    h5_IDs = [ df_ModelSelection_Mets["ModelID"].loc[i] for i in range(len(df_ModelSelection_Mets.index)) if 'H5' in df_ModelSelection_Mets["Hypotheses"].loc[i] ]
    return {'EG': eg_IDs, 'LG':lg_IDs, 'H3a':h3a_IDs, 'H3b':h3b_IDs, 'H3c':h3c_IDs, 'H4a':h4a_IDs, 'H4b':h4b_IDs, 'H5':h5_IDs}

def Model_structure(modelID_breast, modelID_mets):
    import ipywidgets as widgets
    from IPython.display import display
    caption_size = 'h4'
    text1 = widgets.HTMLMath( value=r"<{size}>Let $[B_r]$, $[B_g]$, $[C_r]$, $[C_g]$, $[L_r]$, and $[L_g]$ the number of red and green cells in breast tumor, bloodstream, and lung. The variation of cell population is given by:</{size}>".format(size=caption_size))
    text2 = widgets.HTMLMath( value=r"$$\begin{align} \frac{d[B_r]}{dt} &= \mathcal{G}_{[B_r]} - \lambda [B_r] - \mathcal{I}_{[B_r]}\\ \frac{d[B_g]}{dt} &= \mathcal{G}_{[B_g]} + \lambda [B_r] - \mathcal{I}_{[B_g]}\\ \frac{d[C_r]}{dt} &= \mathcal{I}_{[B_r]} - \phi_{r}[C_r]\\ \frac{d[C_g]}{dt} &= \mathcal{I}_{[B_g]} - \phi_{g}[C_g]\\ \frac{d[L_r]}{dt} &= f_r \phi_{r}[C_r] + \mathcal{G}_{[L_r]}\\ \frac{d[L_g]}{dt} &= f_g \phi_{g}[C_g] + \mathcal{G}_{[L_g]} \end{align}$$")
    # Constraints
    EG_breast = 'EG' in df_ModelSelection_Breast.loc[df_ModelSelection_Breast['ModelID'] == modelID_breast]["Hypotheses"].tolist()[0]
    LG_breast = 'LG' in df_ModelSelection_Breast.loc[df_ModelSelection_Breast['ModelID'] == modelID_breast]["Hypotheses"].tolist()[0]
    NDI = 'NDI' in df_ModelSelection_Breast.loc[df_ModelSelection_Breast['ModelID'] == modelID_breast]["Hypotheses"].tolist()[0]
    DI = 'DI' in df_ModelSelection_Breast.loc[df_ModelSelection_Breast['ModelID'] == modelID_breast]["Hypotheses"].tolist()[0]
    H1 = 'H1' in df_ModelSelection_Breast.loc[df_ModelSelection_Breast['ModelID'] == modelID_breast]["Hypotheses"].tolist()[0]
    H2a = 'H2a' in df_ModelSelection_Breast.loc[df_ModelSelection_Breast['ModelID'] == modelID_breast]["Hypotheses"].tolist()[0]
    H2b = 'H2b' in df_ModelSelection_Breast.loc[df_ModelSelection_Breast['ModelID'] == modelID_breast]["Hypotheses"].tolist()[0]
    EG_lung = 'EG' in df_ModelSelection_Mets.loc[df_ModelSelection_Mets['ModelID'] == modelID_mets]["Hypotheses"].tolist()[0]
    LG_lung = 'LG' in df_ModelSelection_Mets.loc[df_ModelSelection_Mets['ModelID'] == modelID_mets]["Hypotheses"].tolist()[0]
    H3a = 'H3a' in df_ModelSelection_Mets.loc[df_ModelSelection_Mets['ModelID'] == modelID_mets]["Hypotheses"].tolist()[0]
    H3b = 'H3b' in df_ModelSelection_Mets.loc[df_ModelSelection_Mets['ModelID'] == modelID_mets]["Hypotheses"].tolist()[0]
    H3c = 'H3c' in df_ModelSelection_Mets.loc[df_ModelSelection_Mets['ModelID'] == modelID_mets]["Hypotheses"].tolist()[0]
    H4a = 'H4a' in df_ModelSelection_Mets.loc[df_ModelSelection_Mets['ModelID'] == modelID_mets]["Hypotheses"].tolist()[0]
    H4b = 'H4b' in df_ModelSelection_Mets.loc[df_ModelSelection_Mets['ModelID'] == modelID_mets]["Hypotheses"].tolist()[0]
    H5 = 'H5' in df_ModelSelection_Mets.loc[df_ModelSelection_Mets['ModelID'] == modelID_mets]["Hypotheses"].tolist()[0]
    text3 = widgets.HTMLMath( value="Subject to:")
    box_layout = widgets.Layout(display='flex', flex_flow='row', align_items='center', width='100%')
    # Functional constraints
    item_const = []
    item_EG = None; item_LG = None
    if (EG_breast):
        item_EG = 'B_r,B_g'
        if (EG_lung): item_EG = 'B_r,B_g,L_r,Lg'
        else: item_LG = 'L_r,Lg'
    else:
        item_LG = 'B_r,B_g'
        if (LG_lung): item_LG = 'B_r,B_g,L_r,Lg'
        else: item_EG = 'L_r,Lg'
    if (item_EG): item_const.append( widgets.HTMLMath(value=r'$\fbox{EG}: \mathcal{G}_{[i]} = \alpha_{[i]}[i], \quad i=\{'+item_EG+'\}$') )
    if (item_LG): item_const.append( widgets.HTMLMath(value=r'$\fbox{LG}: \mathcal{G}_{[i]} = \alpha_{[i]}[i]\left( 1 - \frac{[i]}{K_{[i]}} \right), \quad i=\{'+item_LG+'\}$') )
    text3_1 = widgets.HBox(children=item_const, layout=box_layout)
    NDI_text = widgets.HTMLMath( value=r'$\fbox{NDI}: \mathcal{I}_{[i]} = \beta_{[i]}[i],  \quad i=\{B_r,B_g \}$')
    DI_text = widgets.HTMLMath( value=r'$\fbox{DI}: \mathcal{I}_{[i]}=\beta_{[i]} \mathcal{H}(t-\tau)[i],  \quad i=\{B_r,B_g \}$')
    if (NDI): text3_2 = NDI_text
    else: text3_2 = DI_text
    # Biological constraints
    item_const = []
    H1_text = widgets.HTMLMath( value=r'$\fbox{H1}: \alpha_{[B_r]} = \alpha_{[B_g]}$')
    H2a_text = widgets.HTMLMath( value=r'$\fbox{H2a}: \beta_{[B_g]} = \beta{[B_r]}$')
    H2b_text = widgets.HTMLMath( value=r'$\fbox{H2b}: \beta_{[B_g]} = 1.5*\beta{[B_r]}$')
    H3a_text = widgets.HTMLMath( value=r'$\fbox{H3a}: \phi_{r} = \phi_{g}$')
    H3b_text = widgets.HTMLMath( value=r'$\fbox{H3b}: \phi_{r} = 3*\phi_{g}$')
    H3c_text = widgets.HTMLMath( value=r'$\fbox{H3c}: \phi_{r} = 6*\phi_{g}$')
    H4a_text = widgets.HTMLMath( value=r'$\fbox{H4a}: f_g\phi_{g} = f_r\phi_{r}$')
    H4b_text = widgets.HTMLMath( value=r'$\fbox{H4b}: f_g\phi_{g} = 1.5*f_r\phi_{r}$')
    H5_text = widgets.HTMLMath( value=r'$\fbox{H5}: \alpha_{[L_r]} = \alpha_{[L_g]}$')
    if (H1): item_const.append(H1_text)
    if (H2a): item_const.append(H2a_text)
    if (H2b): item_const.append(H2b_text)
    if (H3a): item_const.append(H3a_text)
    if (H3b): item_const.append(H3b_text)
    if (H3c): item_const.append(H3c_text)
    if (H4a): item_const.append(H4a_text)
    if (H4b): item_const.append(H4b_text)
    if (H5): item_const.append(H5_text)
    text3_3 = widgets.HBox( children=item_const, layout=box_layout )
    display(text1,text2,text3,text3_1, text3_2,text3_3)

# ODE to PhysiCell
def ScalingODE2PhysiCell():
    LastTime = 40 # days
    dt = 0.005 # 0.005 day = 7.2 min
    N = int(LastTime/dt) + 1
    scaling_factor = 500 #(convertion from ODE scale 1cm^3 to PhysiCell scale 0.002cm^3 = 2 mm^3
    discrete_time = np.linspace(0,LastTime,N,endpoint=True)
    return discrete_time, dt, scaling_factor

def CumulativeCells(InputArray): # Receive cell/min
    import math
    CountTemp = 0
    OutputArray = np.zeros(InputArray.shape)
    for i in range(len(InputArray)):
        DecPart, IntPart = math.modf(InputArray[i]) #  returns the fractional and integer parts of number
        OutputArray[i] = IntPart
        CountTemp += DecPart
        if (CountTemp >= 1): OutputArray[i]+=1; CountTemp = CountTemp - 1.0;
    return OutputArray # return number of cells

def GeneratingDatatoCalibration( modelID_mets,modelID_breast,fileName=None):
    theta_breast = df_ModelSelection_Breast.loc[modelID_breast-1,'MLE Parameter'][0]
    theta_mets = df_ModelSelection_Mets.loc[modelID_mets-1,'MLE Parameter'][0]
    discrete_time, dt, scaling_factor = ScalingODE2PhysiCell()
    output = forward_model_mets_extented(discrete_time,theta_mets, modelID_mets,theta_breast,modelID_breast)
    output /= scaling_factor # RESCALING
    output = np.concatenate((discrete_time.reshape(len(discrete_time),1),output), axis = 1)
    dt_min = dt*1440.0 # dt in minutes
    # print(np.gradient(output[:,7],dt_min).sum()*dt_min,np.gradient(output[:,8],dt_min).sum()*dt_min) # Integral approximation give the total number of cells
    # print(output[-1,7],output[-1,8]) # The last point is equal to the total number of cells
    output[:,0] = output[:,0]*1440.0 # convert days to minutes
    output[:,7] = CumulativeCells(np.gradient(output[:,7])) # discrete cells/min extravasation DsRed (cell^i+1 - cell^i is the number of cells in dt)
    output[:,8] = CumulativeCells(np.gradient(output[:,8])) # discrete cells/min extravasation GFP  (cell^i+1 - cell^i is the number of cells in dt) -> ( [C^{i+1} - C^{i}]/dt_min ) * dt_min -> [C^{i+1} - C^{i}]
    # print(output[:,7].sum(),output[:,8].sum())
    if (fileName):
        np.savetxt(fileName, output, fmt='%1.6e',header='time Breast_RFP+ Breast_GFP+ Intravasation_RFP+ Intravasation_GFP+ Blood_stream_RFP+ Blood_stream_GFP+ Extravasation_rate_int_RFP+ Extravasation_rate_int_GFP+ Lung_RFP+ Lung_GFP+')
    else:
        return output

def model_metastasis_extended(t, y, par, par_breast):
    alpha_br,alpha_bg,K_b,lamb,beta_r,beta_g,tau = par_breast
    alpha_lr,alpha_lg,K_l,phi_r,phi_g,f_Er,f_Eg = par
    # after tau days starts intravasation
    if (t < tau):
        H = 0
    else:
        H = 1
    B_r = alpha_br*y[0]*(1.0 - (y[0]/(K_b - y[1])) ) - lamb*y[0] - H*beta_r*y[0]
    B_g = alpha_bg*y[1]*(1.0 - (y[1]/(K_b - y[0])) ) + lamb*y[0] - H*beta_g*y[1]
    Int_BS_r = H*beta_r*y[0]
    Int_BS_g = H*beta_g*y[1]
    BS_r = H*beta_r*y[0] - phi_r*y[4]
    BS_g = H*beta_g*y[1] - phi_g*y[5]
    Ext_L_r = f_Er*phi_r*y[4]
    Ext_L_g = f_Eg*phi_g*y[5]
    L_r = alpha_lr*y[8]*(1.0 - (y[8]/(K_l - y[9])) ) + f_Er*phi_r*y[4]
    L_g = alpha_lg*y[9]*(1.0 - (y[9]/(K_l - y[8])) ) + f_Eg*phi_g*y[5]

    return [B_r,B_g,Int_BS_r,Int_BS_g,BS_r,BS_g,Ext_L_r,Ext_L_g,L_r,L_g]

def forward_model_mets_extented(times,theta,ID_model,theta_breast,modelID_breast):
    initial_cond = np.array([obs_data.DsRedBreastMean[0],obs_data.GFPBreastMean[0],0.0,0.0,obs_data.DsRedBloodStMean[0],obs_data.GFPBloodStMean[0],0.0,0.0,obs_data.DsRedLungMean[0],obs_data.GFPLungMean[0]],dtype='float64')
    parameters_breast = Apply_Constraints(constraints_breast[modelID_breast-1],parameters_breast_df,theta_breast)
    if ( len(constraints_mets) < ID_model ): ID_model = ID_model%(len(constraints_mets)+1) + 1
    parameter = Apply_Constraints(constraints_mets[ID_model-1],parameters_mets_df,theta)
    output = odeint(model_metastasis_extended, t=times, y0=initial_cond,args=tuple([parameter,parameters_breast]),tfirst=True)
    return output

def Plot_Extended_Model(modelID_mets, modelID_breast, ObsData = True, Plot_rates=False, Scaling = False, log=False, FontSize=None, FileName=None, vert = False):
    theta_breast = df_ModelSelection_Breast.loc[modelID_breast-1,'MLE Parameter'][0]
    theta_mets = df_ModelSelection_Mets.loc[modelID_mets-1,'MLE Parameter'][0]
    discrete_time, dt, scaling_factor = ScalingODE2PhysiCell()
    output = forward_model_mets_extented(discrete_time,theta_mets, modelID_mets,theta_breast,modelID_breast)
    if(vert): fig, (ax1,ax4,ax2,ax5,ax3) = plt.subplots(5,1,figsize=(6, 20))
    else:
        fig, ([[ax1,ax2,ax3],[ax4,ax5,ax6]]) = plt.subplots(2,3,figsize=(18, 8))
        ax6.remove()
    axes = [ax1,ax4,ax2,ax5,ax3]
    if Scaling: output /= scaling_factor # RESCALING
    elif(ObsData):
        # Breast
        ax1.scatter(obs_data.obs_time,obs_data.DsRedBreastMean,color="red")
        ax1.scatter(obs_data.obs_time,obs_data.GFPBreastMean,color="green")
        ax1.errorbar(obs_data.obs_time[1:],obs_data.DsRedBreastMean[1:],yerr=obs_data.DsRedBreastStd[1:],ls='none',ecolor='red',capsize=10)
        ax1.errorbar(obs_data.obs_time[1:],obs_data.GFPBreastMean[1:],yerr=obs_data.GFPBreastStd[1:],ls='none',ecolor='green',capsize=10)
        # Blood stream
        ax2.scatter([obs_data.obs_time[0], obs_data.obs_time[-1]], obs_data.DsRedBloodStMean, color='red')
        ax2.scatter([obs_data.obs_time[0], obs_data.obs_time[-1]], obs_data.GFPBloodStMean, color='green')
        ax2.errorbar(obs_data.obs_time[-1],obs_data.DsRedBloodStMean[-1], yerr=obs_data.DsRedBloodStStd[-1], ls='none',ecolor='red',capsize=10)
        ax2.errorbar(obs_data.obs_time[-1], obs_data.GFPBloodStMean[-1], yerr=obs_data.GFPBloodStStd[-1], ls='none',ecolor='green',capsize=10)
        # Lung
        ax3.scatter(obs_data.obs_time,obs_data.DsRedLungMean,color="red")
        ax3.scatter(obs_data.obs_time,obs_data.GFPLungMean,color="green")
        ax3.errorbar(obs_data.obs_time[1:],obs_data.DsRedLungMean[1:],yerr=obs_data.DsRedLungStd[1:],ls='none',ecolor='red',capsize=10)
        ax3.errorbar(obs_data.obs_time[1:],obs_data.GFPLungMean[1:],yerr=obs_data.GFPLungStd[1:],ls='none',ecolor='green',capsize=10)
    ax1.plot(discrete_time,output[:,0],color="red")
    ax1.plot(discrete_time,output[:,1],color="green")
    ax1.set(xlabel = "Time (days)", ylabel = "Number of cells", title = "Breast tumor")
    ax2.plot(discrete_time,output[:,4],color="red")
    ax2.plot(discrete_time,output[:,5],color="green")
    ax2.set(xlabel = "Time (days)", ylabel = "Number of cells", title = "Bloodstream")
    ax3.plot(discrete_time,output[:,8],color="red")
    ax3.plot(discrete_time,output[:,9],color="green")
    ax3.set(xlabel = "Time (days)", ylabel = "Number of cells", title = "Lung")#,yscale='symlog')
    if (Plot_rates):
        ax4.plot(discrete_time,np.gradient(output[:,2],dt),color="red")
        ax4.plot(discrete_time,np.gradient(output[:,3],dt),color="green")
        ax4.set(xlabel = "Time (days)", ylabel = "Cells/day", title = "Intravasation")
        ax5.plot(discrete_time,np.gradient(output[:,6],dt),color="red")
        ax5.plot(discrete_time,np.gradient(output[:,7],dt),color="green")
        ax5.set(xlabel = "Time (days)", ylabel = "Cells/days", title = "Extravasation")#,yscale='symlog')
    else:
        ax4.plot(discrete_time,output[:,2],color="red")
        ax4.plot(discrete_time,output[:,3],color="green")
        ax4.set(xlabel = "Time (days)", ylabel = "Number of cells", title = "Intravasation")
        ax5.plot(discrete_time,output[:,6],color="red")
        ax5.plot(discrete_time,output[:,7],color="green")
        ax5.set(xlabel = "Time (days)", ylabel = "Number of cells", title = "Extravasation")#,yscale='symlog')

    if(log):
        for ax in axes: ax.set(yscale = 'symlog')
    else:
        for ax in axes:
            yScalarFormatter = ScalarFormatterClass(useMathText=True)
            yScalarFormatter.set_powerlimits((-1,1))
            ax.yaxis.set_major_formatter(yScalarFormatter)
    if FontSize:
        for ax in axes:
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label, ax.yaxis.offsetText] + ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(FontSize)

    plt.subplots_adjust(hspace=0.3)
    fig.tight_layout()
    if FileName: plt.savefig(FileName)
    else: plt.show()

def DisplayExtendedModel(modelID_breast, modelID_mets, Plot_rates, Scaling):
    import ipywidgets as widgets
    from IPython.display import display
    # Model selection outcome
    if (Scaling): display(widgets.HTMLMath(value=r"Scaling tumor from $1~cm^3$ to $2~mm^3$ in 40 days"))
    Plot_Extended_Model(modelID_mets,modelID_breast, Plot_rates=Plot_rates, Scaling=Scaling)

def Plot_CalibPhysiCell(df_PhysiCell,LastDay, dataFile, log=False, FontSize=None, FileName=None, collapse_plot = False, vert = False):
    matrixObs = np.loadtxt(dataFile)
    df_obsOutput = pd.DataFrame({'times': matrixObs[:,0], 'RFP+': matrixObs[:,-2], 'GFP+': matrixObs[:,-1]})
    colorObsDataR = 'k'
    colorObsDataG = 'k'
    if (collapse_plot):
        fig, ax1 = plt.subplots(figsize=(6, 4))
        ax2 = ax1
        colorObsDataR = 'r'
        colorObsDataG = 'g'
    elif(vert): fig, (ax1,ax2) = plt.subplots(2,1,figsize=(6, 8))
    else: fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12, 4))
    df_PhysiCell_local = df_PhysiCell.loc[df_PhysiCell['times'] < 1440.0*LastDay]
    df_obsOutput_local = df_obsOutput.loc[df_obsOutput['times'] < 1440.0*LastDay]
    line_RFP_data, = ax1.plot(df_obsOutput_local['times']/1440.0,df_obsOutput_local['RFP+'],'--',color=colorObsDataR,lw=3)
    line_GFP_data, = ax2.plot(df_obsOutput_local['times']/1440.0,df_obsOutput_local['GFP+'],'--',color=colorObsDataG,lw=3)
    line_RFP, = ax1.plot(df_PhysiCell_local['times']/1440.0,df_PhysiCell_local['RFP_live_cells_mean']+df_PhysiCell_local['RFP_dead_cells_mean'],'-',color='r',label='PhysiCell_RFP+')
    std_RFP = ax1.fill_between(df_PhysiCell_local['times']/1440.0, df_PhysiCell_local['RFP_live_cells_mean']+df_PhysiCell_local['RFP_dead_cells_mean']-df_PhysiCell_local['RFP_live_cells_std']-df_PhysiCell_local['RFP_dead_cells_std'], df_PhysiCell_local['RFP_live_cells_mean']+df_PhysiCell_local['RFP_dead_cells_mean']+df_PhysiCell_local['RFP_live_cells_std']+df_PhysiCell_local['RFP_dead_cells_std'],color='r', alpha=.3)
    line_GFP, = ax2.plot(df_PhysiCell_local['times']/1440.0,df_PhysiCell_local['GFP_live_cells_mean']+df_PhysiCell_local['GFP_dead_cells_mean'],'-',color='g',label='PhysiCell_GFP+')
    std_GFP = ax2.fill_between(df_PhysiCell_local['times']/1440.0, df_PhysiCell_local['GFP_live_cells_mean']+df_PhysiCell_local['GFP_dead_cells_mean']-df_PhysiCell_local['GFP_live_cells_std']-df_PhysiCell_local['GFP_dead_cells_std'], df_PhysiCell_local['GFP_live_cells_mean']+df_PhysiCell_local['GFP_dead_cells_mean']+df_PhysiCell_local['GFP_live_cells_std']+df_PhysiCell_local['GFP_dead_cells_std'],color='g', alpha=.3)
    ax1.set(xlabel='Time (days)', ylabel = 'Number of cells')
    ax2.set(xlabel='Time (days)', ylabel = 'Number of cells')

    if (collapse_plot):
        ax1.legend([line_RFP_data,(line_RFP, std_RFP),line_GFP_data,(line_GFP, std_GFP)],['DsR cells - Macroscale model','DsR cells - Microscale model','GFP cells - Macroscale model','GFP cells - Microscale model'])
        axes = [ax1]
    else:
        ax1.legend([line_RFP_data,(line_RFP, std_RFP)],['DsR cells - Macroscale model','DsR cells - Microscale model'])
        ax2.legend([line_GFP_data,(line_GFP, std_GFP)], ['GFP cells - Macroscale model','GFP cells - Microscale model'])
        axes = [ax1,ax2]

    if(log):
        for ax in axes: ax.set(yscale = 'symlog')
    else:
        for ax in axes:
            yScalarFormatter = ScalarFormatterClass(useMathText=True)
            yScalarFormatter.set_powerlimits((-1,1))
            ax.yaxis.set_major_formatter(yScalarFormatter)

    if FontSize:
        for ax in axes:
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label, ax.yaxis.offsetText] + ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(FontSize)

    plt.subplots_adjust(hspace=0.3)
    fig.tight_layout()
    if FileName: plt.savefig(FileName)
    else: plt.show()

def Plot_CalibPhysiCellKi67(df_replicates, day_ki67, FontSize=None, FileName=None):
    labels = ['DsRed+', 'GFP+']
    colors = ['red', 'lightgreen']
    colors2=['r','g']
    # Quartiles and median from all replicates (boxplot)
    df_replicates_dayKi67 = df_replicates.loc[ df_replicates['times'] == day_ki67*1440 ]
    perc_RFP = 100*df_replicates_dayKi67['RFP_Ki67p_frac']
    perc_GFP = 100*df_replicates_dayKi67['GFP_Ki67p_frac']
    print('Mean(DsRed+): %.2f%%'%perc_RFP.mean(),' Mean(GFP+):  %.2f%%'%perc_GFP.mean())
    fig, ax = plt.subplots(1,1,figsize=(6, 6))
    bplot1 = ax.boxplot([perc_RFP,perc_GFP],  patch_artist=True, labels=labels)
    # fill with colors
    for patch, color in zip(bplot1['boxes'], colors):
        patch.set_facecolor(color)
    for patch in bplot1['medians']:
        patch.set_color('black')
    ax.set(ylim=(None,None),ylabel=('Lung Ki67 (%)'), title='Dist. of Ki67+ cells - day '+str(day_ki67))
    if FontSize:
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label, ax.yaxis.offsetText] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(FontSize)
    if FileName: plt.savefig(FileName)
    else: plt.show()

def Plot_CalibPhysiCellClust(df_Clusters, CellsThreshold = 20, FontSize=None, FileName=None):
    df_Clusters_Threshold = df_Clusters.loc[df_Clusters['NumCells'] >= CellsThreshold]
    df_Clusters_Threshold = df_Clusters_Threshold.sort_values(by=['NumCells'], ignore_index=True)
    df_Cells_RFP_GFP = pd.DataFrame({'GFP': df_Clusters_Threshold['NumCells']*df_Clusters_Threshold['GFP_frac'], 'DsRed': df_Clusters_Threshold['NumCells']*df_Clusters_Threshold['RFP_frac']})
    ax = df_Cells_RFP_GFP.plot(kind='bar',stacked=True,color=['green','red'], figsize=(6, 4), xticks=[])
    ax.set(xlabel='Cluster', ylabel='Number of cells', title='%d Clusters - day 40'%len(df_Clusters_Threshold))
    yScalarFormatter = ScalarFormatterClass(useMathText=True)
    yScalarFormatter.set_powerlimits((-1,1))
    ax.yaxis.set_major_formatter(yScalarFormatter)
    ax.legend(loc='upper left')
    if FontSize:
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label, ax.yaxis.offsetText] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(FontSize)
    if FileName: plt.savefig(FileName)
    else: plt.show()

def latex_float(f):
    float_str = "%1.2e"%f
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str

def GUI_DataViewer():
    import ipywidgets as widgets
    from ipywidgets import fixed
    from IPython.display import display, Math, HTML
    # Randy
    homedir = os.getcwd()
    nanoHUB_flag = False
    if( 'HOME' in os.environ.keys() ):
        nanoHUB_flag = "home/nanohub" in os.environ['HOME']
    # define a Layout giving 50px margin between the items.
    item_layout = widgets.Layout(margin='0 0 50px 0')
    style = """
    <style>
       .jupyter-widgets-output-area .output_scroll {
            height: unset !important;
            border-radius: unset !important;
            -webkit-box-shadow: unset !important;
            box-shadow: unset !important;
        }
        .jupyter-widgets-output-area  {
            height: auto !important;
        }
    </style>
    """
    display(HTML("<style>.container { width:100% !important; }</style>"))
    display(HTML(style))
    about_tab = widgets.Output()
    # define accordion
    Breast_output = widgets.Output(); Mets_output = widgets.Output()
    # internal tabs
    Obs_data_Output = widgets.Output(); Comp_model_Output = widgets.Output(); Parameters_Output = widgets.Output(); ModelSelection_Output = widgets.Output(layout=widgets.Layout(height="auto", width="auto")); Calibration_Output = widgets.Output()
    Obs_data_Output_mets = widgets.Output(); Comp_model_Output_mets = widgets.Output(); Parameters_Output_mets = widgets.Output(); ModelSelection_Output_mets = widgets.Output(layout=widgets.Layout(height="auto", width="auto")); Calibration_Output_mets = widgets.Output(); Macro2Micro_Output = widgets.Output(); M68_Output = widgets.Output(); M89_Output = widgets.Output()
    # define slider
    df_row_best_model = df_ModelSelection_Mets[df_ModelSelection_Mets['weight BIC'] == df_ModelSelection_Mets['weight BIC'].max()]
    SliderModel = widgets.IntSlider(min=1, max=len(df_ModelSelection_Breast.index), step=1, value=df_row_best_model['ModelID breast'], description="Model")
    IntTextResult = widgets.BoundedIntText(min=1, max=len(df_ModelSelection_Breast.index), step=1)
    SliderModel_mets = widgets.IntSlider(min=1, max=len(df_ModelSelection_Mets.index), step=1, value=df_row_best_model['ModelID'], description="Model")
    IntTextResult_mets = widgets.BoundedIntText(min=1, max=len(df_ModelSelection_Mets.index), step=1)
    # link between FloatText and slider
    mylink = widgets.jslink((SliderModel, 'value'), (IntTextResult, 'value'))
    mylink_mets = widgets.jslink((SliderModel_mets, 'value'), (IntTextResult_mets, 'value'))
    # dropbox in calibration
    DropDownPlot = widgets.Dropdown( options=['Solution', 'MCMC Chains', 'Posterior dist'], value='Solution', description='Plot:', disabled=False)
    DropDownPlot_mets = widgets.Dropdown( options=['Solution', 'MCMC Chains', 'Posterior dist'], value='Solution', description='Plot:', disabled=False)
    # radiobuttons in model selection
    RadioButtons_breast = widgets.RadioButtons(options=['AIC','AICc','BIC'], value='BIC',description="Information Criterion:", style = {'description_width': 'initial'})
    RadioButtons_mets = widgets.RadioButtons(options=['AIC','AICc','BIC'], value='BIC',description="Information Criterion:", style = {'description_width': 'initial'})
    hack_radiobuttons_horizontal = HTML('''
    <style>
    .widget-radio-box {
        flex-direction: row !important;
    }
    .widget-radio-box label{
        margin:5px !important;
        width: 120px !important;
    }
    </style>
    ''')
    # check boxes to macro2micro
    IntTextResult_2 = widgets.IntText(min=1, max=len(df_ModelSelection_Breast.index), step=1, disabled=True)
    CheckBox_scaling = widgets.Checkbox(value=True, description='Scaling', disabled=False)
    CheckBox_rates = widgets.Checkbox(value=False, description='Intravasation/Extravasation rates', disabled=False)
    # UI to PhysiCell tab
    DropDown_PhysiCell = widgets.Dropdown( options=['Population', '3D images', 'Clusters', 'Ki67 stain'], value='Population', description='View:', disabled=False)
    CheckBox_logscale = widgets.Checkbox(value=False, description='Log scale', disabled=False)
    slider_3Dimages = widgets.IntSlider(min=0, max=40, step=1.0, value=40, description="Time (days): ")
    IntText_3Dimages = widgets.BoundedIntText(min=0, max=40, step=1.0)
    IntText_defCluster = widgets.BoundedIntText(min=0, max=100, step=1.0,value=100,description="Define cluster (# cells): ", style = {'description_width': 'initial'})
    widgets.jslink((slider_3Dimages, 'value'), (IntText_3Dimages, 'value'))

    def PhysiCell_DropDownFunc(View, log, imageID, list_dfs_calib, list_dfs_clusters, list_dics_images3D, fileObs, CellsdefCluster ):
        ParametersH1 = list_dfs_calib[0]['parameters']
        ParametersH2 = list_dfs_calib[1]['parameters']
        Text_H1 = r"$\textbf{Hypothesis 1:}\quad r_{death} = %s~min^{-1}$"%latex_float(ParametersH1['deadRFP'])
        DsRed_H1 = r"$\text{DsRed+ cells: }r_{01} = %s~min^{-1}$"%latex_float(ParametersH1['prolRFP'])
        GFP_H1 =   r"$\text{GFP+ cells:   }r_{01} = %s~min^{-1}$"%latex_float(ParametersH1['prolGFP'])
        Text_H2 = r"$\textbf{Hypothesis 2:}\quad r_{01} = %s~min^{-1}$"%latex_float(ParametersH2['prolRFP'])
        DsRed_H2 = r"$\text{DsRed+ cells: }r_{death} = %s~min^{-1}$"%latex_float(ParametersH2['deadRFP'])
        GFP_H2 =   r"$\text{GFP+ cells:   }r_{death} = %s~min^{-1}$"%latex_float(ParametersH2['deadGFP'])
        H1_plot = widgets.Output()
        H2_plot = widgets.Output()
        dashboardH1 = widgets.VBox([widgets.HTMLMath(value=DsRed_H1),widgets.HTMLMath(value=GFP_H1),H1_plot],layout=widgets.Layout(border='solid 1px'))
        dashboardH1t = widgets.VBox([widgets.HTMLMath(value=Text_H1),dashboardH1])
        dashboardH2 = widgets.VBox([widgets.HTMLMath(value=DsRed_H2),widgets.HTMLMath(value=GFP_H2),H2_plot],layout=widgets.Layout(border='solid 1px'))
        dashboardH2t = widgets.VBox([widgets.HTMLMath(value=Text_H2),dashboardH2])
        if (View == 'Population'):
            display(CheckBox_logscale)
            with H1_plot: Plot_CalibPhysiCell(list_dfs_calib[0]['stats'], 40, fileObs,log=log,collapse_plot=True)
            with H2_plot: Plot_CalibPhysiCell(list_dfs_calib[1]['stats'], 40, fileObs,log=log,collapse_plot=True)
        if (View == 'Ki67 stain'):
            display(slider_3Dimages)
            with H1_plot: Plot_CalibPhysiCellKi67(list_dfs_calib[0]['raw'],imageID)
            with H2_plot: Plot_CalibPhysiCellKi67(list_dfs_calib[1]['raw'],imageID)
        if (View == 'Clusters'):
            display(IntText_defCluster)
            with H1_plot: Plot_CalibPhysiCellClust(list_dfs_clusters[0],CellsThreshold=CellsdefCluster)
            with H2_plot: Plot_CalibPhysiCellClust(list_dfs_clusters[1],CellsThreshold=CellsdefCluster)
        if (View == '3D images'):
            display(widgets.HBox([slider_3Dimages, IntText_3Dimages]))
            with H1_plot:
                plt.imshow(list_dics_images3D[0][imageID])
                plt.axis('off')
                plt.show()
            with H2_plot:
                plt.imshow(list_dics_images3D[1][imageID])
                plt.axis('off')
                plt.show()
        dashboard = widgets.HBox([dashboardH1t,dashboardH2t])
        display(dashboard)

    def common_filtering():
        with about_tab:
            display(HTML(filename='doc/about.html'))
        with Obs_data_Output:
            Plot_curves_breast()
        with Obs_data_Output_mets:
            Plot_curves_mets(PlotBreast=False)
        with Comp_model_Output:
            ModelsStructures_breast()
        with Comp_model_Output_mets:
            ModelsStructures_mets()
        with Parameters_Output:
            display(parameters_breast_df)
        with Parameters_Output_mets:
            display(parameters_mets_df)
        with ModelSelection_Output:
            PlotSelection(df_ModelSelection_Breast)
            # Plot table of model rank
            Out_Tabs = widgets.interactive_output(TabsSelection, {'df_ModelSelection': fixed(df_ModelSelection_Breast), 'IC': RadioButtons_breast, 'Evidence_Function': fixed(Evidence_Constraints_breast)})
            display(hack_radiobuttons_horizontal,RadioButtons_breast,Out_Tabs)
        with ModelSelection_Output_mets:
            PlotSelection(df_ModelSelection_Mets)
            # Plot table of model rank
            Out_Tabs_mets = widgets.interactive_output(TabsSelection, {'df_ModelSelection': fixed(df_ModelSelection_Mets), 'IC': RadioButtons_mets, 'Evidence_Function': fixed(Evidence_Constraints_mets)})
            display(RadioButtons_mets,Out_Tabs_mets)
        with Calibration_Output:
            outPlotBayes = widgets.interactive_output(PlotResultBayes_breast, {'ModelID': SliderModel, 'Plot': DropDownPlot})
            display(widgets.HBox([SliderModel,IntTextResult,DropDownPlot]))
            display(outPlotBayes)
            outTab = widgets.interactive_output(TabResult_breast, {'ModelID': SliderModel})
            display(outTab)
        with Calibration_Output_mets:
            outPlotBayes_mets = widgets.interactive_output(PlotResultBayes_mets, {'ModelID': SliderModel_mets, 'Plot': DropDownPlot_mets})
            display(widgets.HBox([SliderModel_mets,IntTextResult_mets,DropDownPlot_mets]))
            display(outPlotBayes_mets)
            outTab_mets = widgets.interactive_output(TabResult_mets, {'ModelID': SliderModel_mets})
            display(outTab_mets)
        with Macro2Micro_Output:
            def update_IntTextResult_2_Value(*args):
                IntTextResult_2.value = df_ModelSelection_Mets.loc[df_ModelSelection_Mets['ModelID'] == IntTextResult_mets.value]["ModelID breast"].tolist()[0]
            IntTextResult_mets.observe(update_IntTextResult_2_Value, 'value')
            display(widgets.HBox([widgets.HTMLMath(value="Model ID breast:"), IntTextResult_2, widgets.HTMLMath(value="Model ID mets:"), IntTextResult_mets]))
            df_ModelSelection_Breast.loc[df_ModelSelection_Breast['ModelID'] == modelID_breast]["Hypotheses"].tolist()[0]
            display(widgets.interactive_output(Model_structure, {'modelID_breast': IntTextResult_2, 'modelID_mets': IntTextResult_mets}))
            display(widgets.HBox([CheckBox_scaling,CheckBox_rates]))
            output_macro2micro = widgets.interactive_output(DisplayExtendedModel, {'modelID_breast': IntTextResult_2, 'modelID_mets': IntTextResult_mets, 'Plot_rates': CheckBox_rates, 'Scaling': CheckBox_scaling})
            display(output_macro2micro)
        with M68_Output:
            display(DropDown_PhysiCell)
            list_dfs_calib = [dic_calib_PhysiCell['M68_H1'], dic_calib_PhysiCell['M68_H2']]
            list_dfs_clusters = [dic_cluters_PhysiCell['M68_H1'], dic_cluters_PhysiCell['M68_H2']]
            list_dics_images3D = [dic_images3D_PhysiCell['M68_H1'], dic_images3D_PhysiCell['M68_H2']]
            dashboard = widgets.interactive_output(PhysiCell_DropDownFunc, {'View': DropDown_PhysiCell, 'log': CheckBox_logscale, 'imageID': slider_3Dimages, 'list_dfs_calib': fixed(list_dfs_calib), 'list_dfs_clusters': fixed(list_dfs_clusters), 'list_dics_images3D': fixed(list_dics_images3D), 'fileObs':fixed("data/InSilicoData_B22_M68.dat"), 'CellsdefCluster': IntText_defCluster})
            display(dashboard)
        with M89_Output:
            display(DropDown_PhysiCell)
            list_dfs_calib = [dic_calib_PhysiCell['M89_H1'], dic_calib_PhysiCell['M89_H2']]
            list_dfs_clusters = [dic_cluters_PhysiCell['M89_H1'], dic_cluters_PhysiCell['M89_H2']]
            list_dics_images3D = [dic_images3D_PhysiCell['M89_H1'], dic_images3D_PhysiCell['M89_H2']]
            dashboard = widgets.interactive_output(PhysiCell_DropDownFunc, {'View': DropDown_PhysiCell, 'log': CheckBox_logscale, 'imageID': slider_3Dimages, 'list_dfs_calib': fixed(list_dfs_calib), 'list_dfs_clusters': fixed(list_dfs_clusters), 'list_dics_images3D': fixed(list_dics_images3D), 'fileObs':fixed("data/InSilicoData_B22_M89.dat"), 'CellsdefCluster': IntText_defCluster})
            display(dashboard)

    # create a container for the output with Tabs.
    tabs_breast = widgets.Tab([Obs_data_Output, Comp_model_Output, Parameters_Output, Calibration_Output, ModelSelection_Output],layout=item_layout)
    tabs_breast.set_title(0, 'Observational Data')
    tabs_breast.set_title(1, 'Model Candidates')
    tabs_breast.set_title(2, 'Parameters')
    tabs_breast.set_title(3, 'Calibration')
    tabs_breast.set_title(4, 'Model Selection')

    tabs_mets = widgets.Tab([Obs_data_Output_mets, Comp_model_Output_mets, Parameters_Output_mets, Calibration_Output_mets, ModelSelection_Output_mets],layout=item_layout)
    tabs_mets.set_title(0, 'Observational Data')
    tabs_mets.set_title(1, 'Model Candidates')
    tabs_mets.set_title(2, 'Parameters')
    tabs_mets.set_title(3, 'Calibration')
    tabs_mets.set_title(4, 'Model Selection')

    accordion_Macroscale = widgets.Accordion(children=[tabs_breast, tabs_mets])
    accordion_Macroscale.set_title(0, 'Model Selection Breast')
    accordion_Macroscale.set_title(1, 'Model Selection Metastasis')

    tabs_PhysiCell = widgets.Tab([M68_Output,M89_Output])
    tabs_PhysiCell.set_title(0,'Model 68')
    tabs_PhysiCell.set_title(1,'Model 89')

    # Nesting tabs and accordions
    tab_nest = widgets.Tab(children = [about_tab, accordion_Macroscale, Macro2Micro_Output, tabs_PhysiCell])
    tab_nest.set_title(0, 'About')
    tab_nest.set_title(1, 'Macroscale model')
    tab_nest.set_title(2, 'Macro to Micro')
    tab_nest.set_title(3, 'Microscale model')


    # stack the input widgets and the tab on top of each other with a VBox.
    common_filtering()
    display(tab_nest)

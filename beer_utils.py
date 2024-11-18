# Utilities for the proefpanel dashboard

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

from plotly.express.colors import sample_colorscale
from sklearn.preprocessing import minmax_scale


#######################################################################################
# Data Preps
#######################################################################################

def preprocess(df):
    # Make zoet, zuur, bitter numeric
    df['Sweet'] = df['Sweet'].replace('1 = absent', '1').astype('float')
    df['Bitter'] = df['Bitter'].replace('1 = absent', '1').astype('float')
    df['Sour'] = df['Sour'].replace('1 = absent', '1').astype('float')

    # Make Body, koolzuur, alcohol, kleur, Helderheid numeric
    translate = {'Very little':1., 'Little':2., 'Average':3., 'Much':4., "Very much":5.}
    transcolor = {'Straw':1., 'Yellow':2., 'Golden':3., 'Orange':4., 'Amber':5., 'Copper':6., 'Brown':7.}
    transhelder = {'Turbid':1., 'Cloudy':2., 'Hazy':3., 'Clear':4.}
    transstabiliteit = {'Absent':1., 'Almost absent':2., 'Collapsing':3., 'Stable':4.}
    df.replace({'Body':translate, 
               'Carbonation':translate, 
               'Alcohol':translate, 
               'Color':transcolor, 
               'Clarity':transhelder,
               'Retention':transstabiliteit}, inplace=True)
    df['Date'] = df['Start time'].dt.normalize()

    df.drop(columns=['ID', 'Start time', 'Completion time', 'Email', 'Name', 'Last modified time'], inplace=True)
    df.rename(columns={'Your taster ID':'Taster', 
                      'ID of the beer':'BeerID', 
                      'Which of the following attributes do you encounter':'Attributes', 
                      'Do you have any other feedback?':'Feedback', 
                      'Is this Tripel true to style?':'True-to-style',
                      'Do you think it is fresh or aged?':"Fresh",
                      "Which of the following aromas do you encounter (max 5)?Select the most obvious. Use the \"other\" option if there is one very obvious aroma, that doesn't fit in with the descriptors.":'Aroma'}, inplace=True)

    # Make aroma lists, remove empty items
    df['Aroma'] = df.Aroma.str.split(";").apply(lambda x:x[:-1])
    df['Attributes'] = df.Attributes.astype(str)
    df['Attributes'] = df.Attributes.str.split(";").apply(lambda x:x[:-1])
    
    
    return df

def bieranalyse(df):
    """takes the separate tastings and creates a profile for each beer"""
    df['Feedback'] = df.Feedback + "\n"
    perbier = df.groupby(['BeerID', 'Brewery', 'BeerName', 'Beer']).agg(
        Sweet=('Sweet', 'mean'),
        Bitter=('Bitter', 'mean'),
        Sour=('Sour', 'mean'),
        Carbonation=('Carbonation', 'mean'),
        Alcohol=('Alcohol', 'mean'),
        Clarity=('Clarity', 'mean'),
        Retention=('Retention', 'mean'),
        Aroma=('Aroma', 'sum'),
        Attributes=('Attributes', 'sum'),
        Body=('Body', 'mean'),
        sweetstd=('Sweet', 'std'),
        bitterstd=('Bitter', 'std'),
        sourstd=('Sour', 'std'),
        Feedback=('Feedback', 'sum'),
        n_tasters=('Taster', 'count'),
        Color=('Color', 'mean'),
        pH=('pH', 'mean'),
        ABV=('ABV', 'mean'),
        TFS=('TFS', 'mean'),
        EBC=('EBC', 'mean'),
        AlcoholLabel=('AlcoholLabel', 'mean')
    ).reset_index()
    
    return perbier

def aroma_profiel(aromas):
    """Takes a list of aromas, with duplicate values per vote.
    Returns a df of the number of votes per aroma, normalized to largest.

    Works for Smaakstoffen, too!
    """
    un_ar = list(set(aromas))
    votes = []
    for a in un_ar:
        votes.append(aromas.count(a))

    profiel = pd.DataFrame({'Aroma':un_ar, 'Votes':votes})
    profiel['Votes'] = profiel.Votes
    
    return profiel

def fracinlist(row, threshold=0.25):
    """Helper function to count the aroma/attribute votes,
    compare fraction to threshold and return a list of aromas
    that are above the threshold."""

    counts = {}
    for i in row['Aroma']:
        if i in counts:
            counts[i] += 1/row["Taster"]
        else:
            counts[i] = 1/row["Taster"]
    
    final = [i for i in counts.keys() if counts[i] >= threshold]
    
    return final


#######################################################################################
# Plot Preps
#######################################################################################

def prep_aroma(df, bierdf, topN=15):
    all_aromas = df.Aroma.sum()
    
    # Keep top 15, add any beers that are only in the beer in question
    overall_profiel = aroma_profiel(all_aromas)
    overall_profiel['Votes'] = overall_profiel.Votes / overall_profiel.Votes.max()

    # Bierprofiel
    bier_profiel = aroma_profiel(bierdf.Aroma.values[0])
    bier_profiel['Votes'] = bier_profiel.Votes / bier_profiel.Votes.max()

    bier_profiel.rename(columns={'Votes':'Votes_bier'}, inplace=True)
    
    # Separately determine both topN's and keep those
    toparomas = list(overall_profiel.nlargest(topN, 'Votes').Aroma.values) + \
        list(bier_profiel.nlargest(topN, 'Votes_bier').Aroma.values)
    
    # Samenvoegen en sorteren
    overall_profiel = overall_profiel.merge(bier_profiel, 
                                            how='outer', 
                                            on='Aroma',
                                            ).fillna(0)
    
    overall_profiel = overall_profiel[overall_profiel.Aroma.isin(toparomas)]
    overall_profiel.sort_values(['Votes', 'Votes_bier'], ascending=False, inplace=True)

    return overall_profiel

def colordesc(colornum):
    colornum = int(round(colornum*2))
    coldesc = {2:'straw', 
               3:'straw - yellow',
               4:'yellow', 
               5:'light golden',
               6:'golden', 
               7:'golden - orange',
               8:'orange', 
               9:'light amber',
               10:'amber', 
               11:'dark amber',
               12:'copper', 
               13:'light brown',
               14:'brown'}

    return coldesc[colornum]

#######################################################################################
# Plots
#######################################################################################

def plot_aroma(df):

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df['Aroma'],
        y=df['Votes'],
        name='Overall profile',
        marker_color='lightslategray'
    ))
    fig.add_trace(go.Bar(
        x=df['Aroma'],
        y=df['Votes_bier'],
        name='Beer profile',
        marker_color='indianred',
        opacity=0.8,
        width=0.4
    ))

    # Here we modify the tickangle of the xaxis, resulting in rotated labels.
    fig.update_layout(barmode='overlay', xaxis_tickangle=-45, title="Aroma profile")
    fig.update_layout(legend=dict(
        yanchor="top",
        y=1,
        xanchor="right",
        x=1,
        font_size=16))

    return fig


def plot_basissmaak(df, Beer, switch):

    # Allow to hightlight the bier in question
    df['bierind'] = df.Beer == Beer
    df.sort_values('bierind', inplace=True)

    df['sweetstd'][df.bierind == False] = None
    df['bitterstd'][df.bierind == False] = None

    # df['Sour'] = df.Sour.round(2)
    # df['pH'] = df.pH.astype(str)

    if switch:
        xvar="pH"
        yvar="Sour"
        xerrvar = None
        yerrvar = None
    else:
        xvar="Sweet"
        yvar="Bitter"
        xerrvar = "sweetstd"
        yerrvar = "bitterstd"

    df['dummy_size'] = 1.

    df["pH"][df.pH.isna()] = 5

    fig = px.scatter(df, x=xvar, 
                     y=yvar, 
                     size='dummy_size',
                     size_max=20, 
                     color='bierind', 
                     error_x=xerrvar,
                     error_y=yerrvar,
                     color_discrete_sequence=['lightslategray', 'indianred'],
                     hover_data={'bierind': False, 'Beer': True, 'Sweet': False, 'Bitter':False, 'Sour':False, 'pH':False, 'dummy_size':False},
                     )

    if switch:
        fig.update_layout(xaxis_range=[4,5], yaxis_range=[0.5,2.5])
    else:
        fig.update_layout(xaxis_range=[1,5], yaxis_range=[1,5])
        
    fig.update_layout(showlegend=False)
    


    return fig


def plot_gauge(df, Beer):
    
    fig = go.Figure()
    
    fig.add_trace(go.Indicator(
            mode = "gauge",
            value = df[df.Beer == Beer].Body.values[0],
            domain = {'x': [0, 0.4], 'y': [0.65, 1]},
            title = {'text': "Body"},
            gauge={
                'axis': {'range': [1, 5], 'tickwidth': 1, 'tickcolor': "lightslategray", 
                         'tickvals':[1, 2, 3, 4, 5],
                         'ticktext':['Very little','Little','Average','Much','Very much']},
                'bar': {'color': "indianred"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "lightslategray",
                'steps' : [
                    {'range': [1, 2], 'color': "lightslategray"},
                    {'range': [2, 4], 'color': "white"},
                    {'range': [4, 5], 'color': "lightslategray"}]
                }
        ))
    fig.add_trace(go.Indicator(
            mode = "gauge",
            value = df[df.Beer == Beer].Alcohol.values[0],
            domain = {'x': [0.6, 1], 'y': [0.65, 1]},
            title = {'text': "Alcohol"},
            gauge={
                'axis': {'range': [1, 5], 'tickwidth': 1, 'tickcolor': "lightslategray", 
                         'tickvals':[1.2, 2, 3, 4, 5],
                         'ticktext':['Very little','Little','Average','Much','Very much']},
                'bar': {'color': "indianred"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "lightslategray",
                'steps' : [
                    {'range': [1, 2], 'color': "lightslategray"},
                    {'range': [2, 4], 'color': "white"},
                    {'range': [4, 5], 'color': "lightslategray"}]
                }  
    ))
    
    fig.add_trace(go.Indicator(
            mode = "gauge",
            value = df[df.Beer == Beer].Carbonation.values[0],
            domain = {'x': [0,0.4], 'y': [0, 0.35]},
            title = {'text': "Carbonation"},
            gauge={
                'axis': {'range': [1, 5], 'tickwidth': 1, 'tickcolor': "lightslategray", 
                         'tickvals':[1.2, 2, 3, 4, 5],
                         'ticktext':['Very little','Little','Average','Much','Very much']},
                'bar': {'color': "indianred"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "lightslategray",
                'steps' : [
                    {'range': [1, 2], 'color': "lightslategray"},
                    {'range': [2, 4], 'color': "white"},
                    {'range': [4, 5], 'color': "lightslategray"}]
                }  
    ))
    fig.add_trace(go.Indicator(
            mode = "gauge",
            value = df[df.Beer == Beer].Retention.values[0],
            domain = {'x': [0.6,1], 'y': [0, 0.35]},
            title = {'text': "Head retention"},
            gauge={
                'axis': {'range': [1, 4], 'tickwidth': 1, 'tickcolor': "lightslategray", 
                         'tickvals':[1.2, 2, 3, 4],
                         'ticktext':['Absent','Almost absent','Collapsing','Stable']},
                'bar': {'color': "indianred"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "lightslategray",
                'steps' : [
                    {'range': [1, 2], 'color': "lightslategray"},
                    {'range':[2,3], 'color':'lightgray'},
                    {'range': [3, 4], 'color': "white"}]
                }  
    ))
    fig.add_annotation(x=0.89, y=0.80,
            text=f"ABV: {df[df.Beer == Beer].ABV.values[0]:2.1f}%",
            showarrow=False,
            font={'size':14, 'color':'orange'}
            )
    fig.add_annotation(x=0.89, y=0.73,
            text=f"Label:  {df[df.Beer == Beer].AlcoholLabel.values[0]:2.1f}%",
            showarrow=False,
            font={'size':12, 'color':'orange'}
            )
    fig.add_annotation(x=0.09, y=0.77,
            text=f"TFS: {df[df.Beer == Beer].TFS.values[0]:2.1f} g/L",
            showarrow=False,
            font={'size':14, 'color':'orange'}
            )
    


    return fig

def plot_smaakstoffen(df, nproevers):
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df['Aroma'],
        x=df['Votes']/nproevers,
        # name='Profiel overall',
        marker_color='indianred',
        orientation='h',
        
    ))
    fig.update_layout(title="Attributes", xaxis_range=[0,1])
    fig.update_xaxes(tickformat=".0%", title="Percentage of tasters")

    return fig


def plot_kleur(color):
    fig = go.Figure()

    fig.update_xaxes(range=[0, 1], showgrid=False)
    fig.update_yaxes(range=[0, 1], showgrid=False)

    # Determine color
    colors_ = [1,color,15] 
    discrete_color = sample_colorscale('solar_r', minmax_scale(colors_))[1]

    fig.add_shape(type="rect",
        x0=0, y0=0, x1=1, y1=1,
        line=dict(
            color="lightslategray",
            width=2,
        ),
        fillcolor=discrete_color, 
        )
    
    fig.update_layout(title='Color of the beer',
                      yaxis_showticklabels=False,
                      xaxis_showticklabels=False)

    return fig


def plot_tasters_tastes(df, Taster, taste="Sweet"):

    df_taster = df[df.Taster == Taster]
    
    mean_taster = df_taster[taste+"_err"].mean()

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df[taste+'_err'],
        marker_color='lightslategrey',
        autobinx=False,
        xbins=dict(
            start=-2.5,
            end=2.5,
            size=0.5
        ),
        name="All tastings",
        histnorm=None #"count"    
    ))
    fig.add_vline(x=0, line_width=5)
    fig.add_trace(go.Histogram(
        x=df_taster[taste+'_err'],
        marker_color='indianred',
        autobinx=False,
        xbins=dict(
            start=-2.5,
            end=2.5,
            size=0.5
        ),
        opacity=0.75,
        name="Selected Taster",
        histnorm=None #'count'    
    ))
    fig.update_layout(barmode='overlay')
    fig.add_vline(x=mean_taster, line_width=3, line_color='indianred')
    

    fig.update_layout(title=taste+"ness perception sensitivity", 
                      xaxis_range=[-2.5,2.5],
                      xaxis_title="Difference taster - average",
                      yaxis_title="Number")
    
    return fig

def plot_sensitivity(df, Taster, col='Aroma', threshold = 0.25):
    """Plot to visualize teh sensitivity of Taster to the various
    attributes in col (probably either Aroma or Attributes).
    
    There will be a threshold set to determine presence in a beer:
    at least a factor threshold of all tasters need to indicate this
    aroma/attribute for it to be "in the beer". Then, for all of those
    beers, the sensitivity of Taster is the fraction of times Taster recognized 
    it, only from beers he/she had him/herself.

    The threshold may become a variable in the dashboard.    
    """

    # Determine the beers this tasters has had and filter df on it.
    beers = np.unique(df[df.Taster == Taster].Beer)
    df_relevant = df[df.Beer.isin(beers)]

    # Determine which beers have which aromas/attributes
    pb = df_relevant.groupby("Beer")
    aromas = pd.DataFrame(pb.Aroma.sum()).reset_index()
    ntastings = pd.DataFrame(pb.Taster.count()).reset_index()
    aromas = aromas.merge(ntastings, on="Beer")
    aromas['Aromas'] = aromas.apply(fracinlist, threshold=threshold, axis=1)
    aromas.drop(columns=["Aroma", "Taster"], inplace=True)
    
    df_taster = df_relevant[df_relevant.Taster == Taster].merge(aromas, on="Beer")
    
    # Stupidly loop over all Aromas
    all_aromas = np.unique(df_taster.Aromas.sum())

    aromadict = {}
    present = df_taster.Aromas.values
    tasted = df_taster.Aroma.values
    for ar in all_aromas:
        aromadict[ar] = 0
        nbeers = 0
        for p, t in zip(present, tasted):
            if ar in p:
                nbeers += 1
                if ar in t: aromadict[ar] += 1
        aromadict[ar] /= nbeers

    df_plot = pd.DataFrame.from_dict(aromadict, orient="index", columns=["Sensitivity"]
                                     ).sort_values('Sensitivity', ascending=False).reset_index()

    # Plot for all
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_plot['index'],
        y=df_plot['Sensitivity'],
        marker_color='indianred'
    ))
    fig.update_layout(title=f"Sensitivity for aromas, threshold: {threshold:.2f}")
    fig.update_xaxes(title="Aroma")
    fig.update_yaxes(title="Sensitivity")

    return fig



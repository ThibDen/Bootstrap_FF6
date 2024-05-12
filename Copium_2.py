import pandas as pd
import numpy as np
import pandas_datareader as pdr
from numba import jit
import matplotlib.pyplot as plt
import imageio
import io
from IPython.display import Image, display
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
#Code written by Thib Den, please send me if on RR if you find any bugs/mistakes in the code


#General inputs 
months=240 #Number of Months you want to simualte
nr_sim=1000 #Number of simulations you want to simualte

Start_invest=1000 #Invest amount start
Monthly_DCA=1000 #Monthly contributions
Transaction_costs=0.42 #% costs per buy/sell, for example, in Belgium 0.12% tax + 3 euro broker costs if I DCA 1000 is aprox 0.42. This cost is also used for rebalancing and for initial investment so 0.42 would for me probably be an overstatement.
rebalance=12 #Per how many months are you going to rebalance your portfolio

haircut_min=0.5
haircut_max=1

min_bootstrap=60 #How long of a sample do you want to draw, 
max_bootstrap=120

'''
Percent_portfolio: target portfolio percentage of each ETF: this has to sum up to one, RF bonds can be incorporated by putting all factors to zeo
Market: on which market did you estimate the Factor loadings for that ETF:
    -"DEV": Developped 
    -"EM": Emerging
    -"US": United States of America
    -"EU": Europe
#Factor_loadings= Exposure to risk factors in order ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA','WML']
Monthly_alfa: Excess monthly returns of portfolio in %
ETF_tracking_costs_yearly: Total yearly in % costs for portfolio (TER + implicit trading costs/divident leakage)

 '''
#Portfolio A
portfolio_data_A = [
    {"ETF": "JPGL", "Percent_portfolio": 50, "Market": 'US', "Factor_loadings": [0.87763387, 0.01803864, 0.055651  , 0.1827293 , 0.12556687,    0.00421593, 0.87763387],'Monthly_alfa':0.071456,'ETF_tracking_costs_yearly': 0.84},
    {"ETF": "ZPRX", "Percent_portfolio": 17, "Market": 'US', "Factor_loadings": [ 1.09549288,  0.22298114,  0.17133497,  0.11936945,  0.175671  ,-0.25264292,  1.09549288],'Monthly_alfa':0.05398,'ETF_tracking_costs_yearly': 0.7},
    {"ETF": "ZPRV", "Percent_portfolio": 33, "Market": 'US', "Factor_loadings": [ 1.03430705,  0.77306623,  0.29128111,  0.19899231,  0.08163184,-0.17861488,  1.03430705],'Monthly_alfa':0.0821,'ETF_tracking_costs_yearly': 0.65},
]


#Portfolio B
portfolio_data_B = [
    {"ETF": "IFSW", "Percent_portfolio": 100, "Market": 'US', "Factor_loadings": [0.95598271, 0.09530343, 0.06249563, 0.12509891, 0.00394935,  0.0833609, 0.95598271],'Monthly_alfa':0.0006160213093202313,'ETF_tracking_costs_yearly': 0.75},
#     {"ETF": "IS3R", "Percent_portfolio": 34, "Market": 'US', "Factor_loadings": [0.98,-0.04,0.6,0.15,-0.26,0.43,1],'Monthly_alfa':-0.09,'ETF_tracking_costs_yearly': 0.54},
#    {"ETF": "ZPRV", "Percent_portfolio": 44, "Market": 'US', "Factor_loadings": [1.14,0.67,0.24,0.01,0.28,-0.29,1],'Monthly_alfa':0.28,'ETF_tracking_costs_yearly': 0.65},
#   {"ETF": "ZPRX", "Percent_portfolio": 22, "Market": 'US', "Factor_loadings": [1.3,0.73,0.42,0.3,0.03,-0.26,1],'Monthly_alfa':-0.07,'ETF_tracking_costs_yearly': 0.7},
#
]


# The rest of the code is where the magic happens

def Portfolio_values(portfolio_data):
    Portfolio_dist=np.array([etf['Percent_portfolio'] for etf in portfolio_data])/100
    Markets = np.array([etf['Market'] for etf in portfolio_data])
    Factor_loadings=np.array([etf['Factor_loadings'] for etf in portfolio_data])
    Monthly_alfa_min_cost=np.array([etf['Monthly_alfa'] for etf in portfolio_data])/100-(np.power(1+np.array([etf['ETF_tracking_costs_yearly'] for etf in portfolio_data])/100,1/12)-1)
    return Portfolio_dist,Markets,Factor_loadings,Monthly_alfa_min_cost

def get_factors(name1,name2, start_date, end_date):
    factors_5F = pdr.DataReader(
        name=name1,
        data_source="famafrench", 
        start=start_date, 
        end=end_date)[0]

    factor_MOM = pdr.DataReader(
        name=name2,
        data_source="famafrench", 
        start=start_date, 
        end=end_date)[0]    
    factors = pd.concat([factors_5F, factor_MOM], axis=1)
    cols = [col for col in factors.columns if col != 'RF']
    factors = factors[cols + ['RF']]

    if 'Mom   ' in factors.columns:
        factors = factors.rename(columns={'Mom   ': 'WML'})
    factors = factors.dropna(subset=['WML'])
    factors = factors.dropna(subset=['Mkt-RF'])
    return (factors
        .divide(100)
        .reset_index()
        .rename(columns={"index": "Date"})
        .assign(Date=lambda x: pd.to_datetime(x["Date"].astype(str))))

start_date = "1960-01-01"
end_date = pd.to_datetime('today').normalize()
factors_ff6_DEV = get_factors("Developed_5_Factors","Developed_Mom_Factor", start_date, end_date)
factors_ff6_US = get_factors("F-F_Research_Data_5_Factors_2x3","F-F_Momentum_Factor", start_date, end_date)
factors_ff6_EU = get_factors("Europe_5_Factors","Europe_Mom_Factor",start_date, end_date)
factors_ff6_EM = get_factors("Emerging_5_Factors","Emerging_Mom_Factor", start_date, end_date)





def rename_columns(df, suffix):
    # Exclude 'Date' column from renaming
    columns = [Rf"{col}_{suffix}" if col != 'Date' else col for col in df.columns]
    df.columns = columns
    return df

# Rename the columns with the appropriate suffixes
factors_ff6_DEV = rename_columns(factors_ff6_DEV, 'dev')
factors_ff6_US = rename_columns(factors_ff6_US, 'us')
factors_ff6_EU = rename_columns(factors_ff6_EU, 'eu')
factors_ff6_EM = rename_columns(factors_ff6_EM, 'em')


# Then, merge the datasets on the 'Date' column
merged_data = pd.merge(factors_ff6_DEV, factors_ff6_US, on='Date')
merged_data = pd.merge(merged_data, factors_ff6_EU, on='Date')
merged_data = pd.merge(merged_data, factors_ff6_EM, on='Date')




def process_data(merged_data,Markets, factors_ff6_US,Tot_fact_exposure,Factor_loadings ):
    if np.sum(Markets=='EM') !=0:
        merged_data = merged_data.loc[merged_data.Date > "1992-06-01"]
        merged_data_np = merged_data.to_numpy()
        merged_data_np = np.array(merged_data_np[:, 1:], dtype=np.float64)

    elif np.sum(Markets!='US')==0: #If only US make use of longer time horzion, and add inflation
        US_cpi=pd.read_csv("CPI_US.csv",parse_dates=True)
        US_cpi['DATE'] = pd.to_datetime(US_cpi['DATE'])      
        US_cpi = US_cpi.rename(columns={'DATE': 'Date'})
        US_cpi['CPILFESL']=US_cpi['CPILFESL'].pct_change()
        merged_data= pd.merge(factors_ff6_US,US_cpi, on='Date') 

        merged_data_np = merged_data.to_numpy()
        merged_data_np = np.array(merged_data_np[:, 1:], dtype=np.float64)
        Tot_fact_exposure = np.c_[Factor_loadings, -np.ones(Factor_loadings.shape[0])] #add inflation loading
    else:
        merged_data_np = merged_data.to_numpy()
        merged_data_np = np.array(merged_data_np[:, 1:], dtype=np.float64)

    return merged_data_np, Tot_fact_exposure

# Call the function


Net_of_costs=1-Transaction_costs/100

@jit(nopython=True)
def calculate_values(months, Data,Tot_fact_exposure,nr_sim,start_cap, monthly_cap, monthly_correction,Portfolio_dist,Net_of_costs,rebalance, haircut_min, haircut_max):
    Value = np.zeros((nr_sim ,np.shape(Tot_fact_exposure)[0],months))
    Value_fixed_inv = np.zeros((nr_sim ,np.shape(Tot_fact_exposure)[0],months))


    if np.shape(Tot_fact_exposure)[1]==8:
        nr_markets= 1
    else:
        nr_markets=4
    nr_etf=np.shape(Tot_fact_exposure)[0]
    for k in range( nr_sim):
        lenght = np.random.randint(min_bootstrap, max_bootstrap+1, size=int(months/min_bootstrap))
        start = np.empty_like(lenght)

        for i in range(len(lenght)):
            start[i] = np.random.randint(0, np.shape(Data)[0]-lenght[i])
        indices = []
        for a, b in zip(lenght, start):
            indices.extend(np.arange(b, b+a))
        indices = np.array(indices)[:months]
        for i in range(nr_markets):
            Tot_fact_exposure[:,7*i+1:7*(i+1)-1]=Tot_fact_exposure[:,7*i+1:7*(i+1)-1]*(1-np.random.uniform(haircut_min, haircut_max,size=(nr_etf,5)))
        returns= 1+np.dot(Data[indices,:], Tot_fact_exposure.T)+monthly_correction
        Value[k,:,0] = start_cap*Portfolio_dist*Net_of_costs
        Value_fixed_inv[k,:,0]=start_cap*Portfolio_dist*Net_of_costs
        for i in range(1,months):
            Value[k,:,i]  = Value[k,:,i-1] *returns[i,:]
            Value_fixed_inv[k,:,i]  = Value_fixed_inv[k,:,i-1] *returns[i,:]
            Value[k,:,i]= Value[k,:,i]+(Value[k,:,i]/np.sum(Value[k,:,i])-Portfolio_dist==np.min(Value[k,:,i]/np.sum(Value[k,:,i])-Portfolio_dist) )*monthly_cap*Net_of_costs
            if i % rebalance == 0:
                Value[k,:,i]=Value[k,:,i]-((Value[k,:,i]-Portfolio_dist*np.sum(Value[k,:,i]))>0)*(Value[k,:,i]-Portfolio_dist*np.sum(Value[k,:,i]))/Net_of_costs
                Value[k,:,i]=Value[k,:,i]-((Value[k,:,i]-Portfolio_dist*np.sum(Value[k,:,i]))<0)*(Value[k,:,i]-Portfolio_dist*np.sum(Value[k,:,i]))*Net_of_costs
                Value_fixed_inv[k,:,i]=Value_fixed_inv[k,:,i]-((Value_fixed_inv[k,:,i]-Portfolio_dist*np.sum(Value_fixed_inv[k,:,i]))>0)*(Value_fixed_inv[k,:,i]-Portfolio_dist*np.sum(Value_fixed_inv[k,:,i]))/Net_of_costs
                Value_fixed_inv[k,:,i]=Value_fixed_inv[k,:,i]-((Value_fixed_inv[k,:,i]-Portfolio_dist*np.sum(Value_fixed_inv[k,:,i]))<0)*(Value_fixed_inv[k,:,i]-Portfolio_dist*np.sum(Value_fixed_inv[k,:,i]))*Net_of_costs

    return np.sum(Value,axis=1), np.sum(Value_fixed_inv,axis=1)

# Call the optimized function

def create_histogram_gif(Returns_A, Returns_B, Start_invest, Monthly_DCA, portfolio_name_A, portfolio_name_B):
    # Create a list to store each frame of the GIF
    images = []
    indic=np.mean(Returns_A)
    if indic>2:
    # Define the bins for the histogram
        bins = np.linspace(min(np.percentile(Returns_A[:,-1], 0.01),np.percentile(Returns_B[:,-1], 0.01)), max(np.percentile(Returns_A[:,-1], 99),np.percentile(Returns_B[:,-1], 99)), 50)
    else:
        bins = np.linspace(0.85,1.30, 50)


    # Define the range for the y-axis (you can adjust these values as needed)
    y_min = 0

    # Loop over each year (assuming Returns is monthly data)
    for i in range(11, Returns_A.shape[1], 12):
        if indic>2:
            y_max=max(np.shape(Returns_A)[0]*(1-(2*(i-11)/np.shape(Returns_A)[1])),np.shape(Returns_A)[0]/4)
        else:
            y_max=np.shape(Returns_A)[0]/4
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create histogram with fixed bins for Returns_A
        ax.hist(Returns_A[:, i], bins=bins, color='blue', alpha=0.5, label=f'{portfolio_name_A}')

        # Create histogram with fixed bins for Returns_B
        ax.hist(Returns_B[:, i], bins=bins, color='red', alpha=0.5, label=f'{portfolio_name_B}')

        ax.set_title(f'DCA of {Monthly_DCA} after {i//12+1} years, haircut range of [{haircut_min}, {haircut_max}]', fontsize=16)
        ax.set_xlabel('Returns (%)')
        ax.set_ylabel('Frequency')
        ax.set_ylim(y_min, y_max)
        ax.legend()

        # Save plot to a PNG file in memory
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        # Add PNG file to images list
        images.append(imageio.imread(buf))

        # Close the figure
        plt.close()

    # Create GIF from images list
    gif = imageio.mimwrite(imageio.RETURN_BYTES, images, 'GIF', duration=0.5)

    # Display GIF in Jupyter Notebook
    display(Image(data=gif))

def Table(data, title_port, table_text):
    percentiles = [1, 10, 25, 50, 75, 90, 99]
    labels = ['Worst 1%', 'Worst 10%', 'Worst 25%', 'Median', 'Best 25%', 'Best 10%', 'Best 1%']
    years = [1, 3, 5, 10, 15, 20]

    # Convert years to months
    months = [year * 12-1 for year in years]

    # Initialize an empty dataframe to store the results
    df = pd.DataFrame(index=labels)
    # Calculate the percentiles for each year
    for month in months:
        percentile_values = np.percentile(data[:, month], percentiles, axis=0)
        df[f'After {month//12+1} years'] = percentile_values

    # Define the title
    print(f'{title_port}: {table_text}')

    # Apply the float format to each cell in the DataFrame
    if df.iloc[1,1]>3:
    # Print the DataFrame in a nice table format
        print(df.to_markdown(floatfmt='.2f'))
    else:
        print(df.to_markdown(floatfmt='.4f'))
  


# Call the function
def calculate_returns_and_plot(months, merged_data, nr_sim, Start_invest, Monthly_DCA,Net_of_costs,rebalance, portfolio_data_A,portfolio_data_B, haircut_min, haircut_max):

    Portfolio_dist_A,Markets_A,Factor_loadings_A,Monthly_alfa_min_cost_A=Portfolio_values(portfolio_data_A)
    Portfolio_dist_B,Markets_B,Factor_loadings_B,Monthly_alfa_min_cost_B=Portfolio_values(portfolio_data_B)

    Tot_fact_exposure_A = np.repeat(np.transpose(np.array((['DEV'],['US'],['EU'],['EM']))==Markets_A), Factor_loadings_A.shape[1], axis=1)*np.tile(Factor_loadings_A, (1,4))
    Tot_fact_exposure_B = np.repeat(np.transpose(np.array((['DEV'],['US'],['EU'],['EM']))==Markets_B), Factor_loadings_B.shape[1], axis=1)*np.tile(Factor_loadings_B, (1,4))


    merged_data_np_A, Tot_fact_exposure_A = process_data(merged_data,Markets_A, factors_ff6_US, Tot_fact_exposure_A ,Factor_loadings_A)
    merged_data_np_B, Tot_fact_exposure_B = process_data(merged_data,Markets_B, factors_ff6_US, Tot_fact_exposure_B ,Factor_loadings_B)

    Tot_fact_exposure_A=np.array(Tot_fact_exposure_A, dtype=np.float64)
    Tot_fact_exposure_B=np.array(Tot_fact_exposure_B, dtype=np.float64)

    Value_A, Value_fixed_inv_A = calculate_values(months, merged_data_np_A, Tot_fact_exposure_A, nr_sim, Start_invest, Monthly_DCA, Monthly_alfa_min_cost_A,Portfolio_dist_A,Net_of_costs,rebalance, haircut_min, haircut_max)
    Value_B, Value_fixed_inv_B = calculate_values(months, merged_data_np_B, Tot_fact_exposure_B, nr_sim, Start_invest, Monthly_DCA, Monthly_alfa_min_cost_B,Portfolio_dist_B,Net_of_costs,rebalance, haircut_min, haircut_max)

    IRR_A=np.power(Value_fixed_inv_A/Start_invest,12/np.arange(1,months+1))
    IRR_B=np.power(Value_fixed_inv_B/Start_invest,12/np.arange(1,months+1))

    title_port_A =  f'{"/".join([d["ETF"] for d in portfolio_data_A])}:{"/".join([str(d["Percent_portfolio"]) for d in portfolio_data_A])}'
    title_port_B =  f'{"/".join([d["ETF"] for d in portfolio_data_B])}:{"/".join([str(d["Percent_portfolio"]) for d in portfolio_data_B])}'

    invested = Start_invest + Monthly_DCA * np.arange(months)

    percentiles = [1, 10, 25,50, 75, 90, 99]
    labels = ['Worst 1%', 'Worst 10%', 'Worst 25%', 'Median','Best 25%', 'Best 10%', 'Best 1%']
    colors = ['black', 'red', 'orange', 'Blue','lime', 'green', 'darkgreen']

    returns_A = [(np.nanpercentile(Value_A, p, axis=0) - invested) / invested for p in percentiles]
    returns_B = [(np.nanpercentile(Value_B, p, axis=0) - invested) / invested for p in percentiles]

    chance_of_profit_A = np.nansum(Value_A > invested, axis=0) / (np.shape(Value_A)[0] - np.nansum(np.isnan(Value_A), axis=0))
    chance_of_profit_B = np.nansum(Value_B > invested, axis=0) / (np.shape(Value_B)[0] - np.nansum(np.isnan(Value_B), axis=0))

    # Prepend initial_point to each time series
    returns_A,returns_B = [np.insert(r, 0, 0) for r in returns_A], [np.insert(r, 0, 0) for r in returns_B]

    year_t = np.arange(len(returns_A[0])) / 12  # Assuming 'months' represents the time in months

# Calculate the maximum of the 99th percentile of Returns_A and Returns_B
    y_max = max(np.percentile(Value_A[:,-1]/invested[-1], 99),np.percentile(Value_B[:,-1]/invested[-1], 99)) * 100

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 14))

    # Plot for returns_A
    for r, label, color in zip(returns_A, labels, colors):
        ax1.plot(year_t, r * 100, label=label, color=color)
    ax1.set_xlabel('Years')
    ax1.set_ylabel('Returns on money invested(%)')
    ax1.set_title(f"{title_port_A}")
    ax1.legend()
    ax1.grid(True)
    ax1.set_ylim([-100, y_max])  # Set the y-axis limit

    # Plot for returns_B
    for r, label, color in zip(returns_B, labels, colors):
        ax2.plot(year_t, r * 100, label=label, color=color)
    ax2.set_xlabel('Years')
    ax2.set_title(f"{title_port_B}")
    ax2.legend()
    ax2.grid(True)
    ax2.set_ylim([-100, y_max])  # Set the y-axis limit

    fig.suptitle(f'Returns with DCA of {Monthly_DCA}, haircut range of [{haircut_min}, {haircut_max}]', fontsize=16)

    plt.show()


    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot for chance_of_profit_A
    ax.plot(year_t[1:], chance_of_profit_A * 100, label=title_port_A, color='blue')

    # Plot for chance_of_profit_B
    ax.plot(year_t[1:], chance_of_profit_B * 100, label=title_port_B, color='red')

    ax.set_xlabel('Years')
    ax.set_ylabel('Chance of profit (%)')
    ax.set_title(f'Probability of profit with DCA of {Monthly_DCA}, haircut range of [{haircut_min}, {haircut_max}]', fontsize=16)
    ax.legend()
    ax.grid(True)

    plt.show()
    create_histogram_gif((Value_A/invested-1)*100, (Value_B/invested-1)*100, Start_invest, Monthly_DCA, title_port_A, title_port_B)
    create_histogram_gif(IRR_A, IRR_B, 0, 1, title_port_A, title_port_B)
    table_text=f'End value with DCA of {Monthly_DCA}, net of inlation'
    Table(Value_A, title_port_A, table_text)
    Table(Value_B, title_port_B, table_text)
    table_text= 'Anualized returns, net of inflation'
    Table(IRR_A, title_port_A, table_text)
    Table(IRR_B, title_port_B, table_text)
# Call the function

calculate_returns_and_plot(months, merged_data, nr_sim, Start_invest, Monthly_DCA,Net_of_costs,rebalance, portfolio_data_A,portfolio_data_B, haircut_min, haircut_max)


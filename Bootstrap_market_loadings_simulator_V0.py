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

start_date = "1960-01-01"
end_date = pd.to_datetime('today').normalize()


#General inputs 
months=240 #Number of Months you want to simualte
nr_sim=100000 #Number of simulations you want to simualte

Start_invest=1000 #Invest amount start
Monthly_DCA=1000 #Monthly contributions
Transaction_costs=0.3 #% costs per buy/sell, for example, in Belgium 0.12% tax +0.18%  broker costs This cost is also used for rebalancing and for initial investment so 0.30 would for me probably be an overstatement.

haircut_min=0 #minimum haircut we want to apply, on the 5 factors 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA','WML'
haircut_max=0.5 #maximum haircut

min_bootstrap=60 #How long of a sample do you want to draw, at minumum
max_bootstrap=120 #How long of a sample do you want to draw, at maximum

#Portfolio A
monthly_alpha_A=0.0 #Excess yearly returns of portfolio in %
yearly_portfolio_cost_A=0.7 #Total yearly in % costs for portfolio (TER + implicit trading costs/divident leakage)
factor_exposure_Dev_A=np.array([1,0.5,0.5,0.5,0.5,0.5,0])*0.88 #Exposure to risk F-F factors of the developped world: in order ['Date', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA','RF']
factor_exposure_US_A=np.array([1,0.5,0.5,0.5,0.5,0.5,1])*0#Exposure to risk F-F factors of the US: in order ['Date', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA','RF']
factor_exposure_EU_A=np.array([1,0.5,0.5,0.5,0.5,0.5,0])*0 #Exposure to risk F-F factors of the EU: in order ['Date', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA','RF']
factor_exposure_EM_A=np.array([1,0.5,0.5,0.5,0.5,0.5,0])*0.12 #Exposure to risk F-F factors of the emerging market: in order ['Date', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA','RF']

#Portfolio B
monthly_alpha_B=0
yearly_portfolio_cost_B=0.4
factor_exposure_Dev_B=np.array([1,0,0,0,0,0,0])*0.88
factor_exposure_US_B=np.array([1,0.4,0.4,0.4,0.4,0.4,1])*0
factor_exposure_EU_B=np.array([0,0,0,0,0,0,0])*0
factor_exposure_EM_B=np.array([1,0,0,0,0,0,0])*0.12


#DATA RETRIEVAL
Tot_fact_exposure_A=np.concatenate((factor_exposure_Dev_A,factor_exposure_US_A,factor_exposure_EU_A,factor_exposure_EM_A))
Tot_fact_exposure_B=np.concatenate((factor_exposure_Dev_B,factor_exposure_US_B,factor_exposure_EU_B,factor_exposure_EM_B))
Monthly_alfa_min_cost_A= monthly_alpha_A/100 + (1- yearly_portfolio_cost_A/100)**(1/12)-1
Monthly_alfa_min_cost_B= monthly_alpha_B/100+ (1- yearly_portfolio_cost_B/100)**(1/12)-1
def get_factors(name1,name2, start_date, end_date): #this code retrieves the factors from Keneth French site. 
    factors_5F = pdr.DataReader( #5 factors
        name=name1,
        data_source="famafrench", 
        start=start_date, 
        end=end_date)[0]

    factor_MOM = pdr.DataReader( #also retrieve momentum
        name=name2,
        data_source="famafrench", 
        start=start_date, 
        end=end_date)[0]    
    factors = pd.concat([factors_5F, factor_MOM], axis=1) #merge momentum with the other 5 factors
    cols = [col for col in factors.columns if col != 'RF'] #Switch the collumns order such that RF is the last (makes applying the haircut easier)
    factors = factors[cols + ['RF']] #Switch the collumns order such that RF is the last (makes applying the haircut easier)

    if 'Mom   ' in factors.columns:
        factors = factors.rename(columns={'Mom   ': 'WML'}) #In US data, the WML factor is called Mom, switch the name. 
    factors = factors.dropna(subset=['WML']) #drop observations with missing WML (in DEV, EM and EU, WML data starts a bit later)
    factors = factors.dropna(subset=['Mkt-RF'])#drop observations with missing 5 factors (in US,  WML data starts a bit earlier)
    return (factors
        .divide(100) #divide it by 100 to get form % to absolute values)
        .reset_index()
        .rename(columns={"index": "Date"})
        .assign(Date=lambda x: pd.to_datetime(x["Date"].astype(str)))) #set the date collumns in uniform format for merging. 

start_date = "1960-01-01" #if you want to limit the date range from where we draw the factors
end_date = pd.to_datetime('today').normalize() #if you want to limit the date range from where we draw the factors

#Download all factor data
factors_ff6_DEV = get_factors("Developed_5_Factors","Developed_Mom_Factor", start_date, end_date)
factors_ff6_US = get_factors("F-F_Research_Data_5_Factors_2x3","F-F_Momentum_Factor", start_date, end_date)
factors_ff6_EU = get_factors("Europe_5_Factors","Europe_Mom_Factor",start_date, end_date)
factors_ff6_EM = get_factors("Emerging_5_Factors","Emerging_Mom_Factor", start_date, end_date)

def rename_columns(df, suffix):
    # Exclude 'Date' column from renaming
    columns = [f"{col}_{suffix}" if col != 'Date' else col for col in df.columns]
    df.columns = columns
    return df

# Rename the columns with the appropriate suffixes
factors_ff6_DEV = rename_columns(factors_ff6_DEV, 'dev')
factors_ff6_US = rename_columns(factors_ff6_US, 'us')
factors_ff6_EU = rename_columns(factors_ff6_EU, 'eu')
factors_ff6_EM = rename_columns(factors_ff6_EM, 'em')


# Then, merge the datasets on the 'Date' column, this is to get one dataset with al factors. 
merged_data = pd.merge(factors_ff6_DEV, factors_ff6_US, on='Date')
merged_data = pd.merge(merged_data, factors_ff6_EU, on='Date')
merged_data = pd.merge(merged_data, factors_ff6_EM, on='Date')






#this is to select the right dataset depending on the market.
def process_data(merged_data, factors_ff6_US,Tot_fact_exposure ):
    if np.sum(Tot_fact_exposure[21:28]==0)< 7: #If there are emerging markets, start from 1992-06, beforehand some factors are missing. (coded as -0.999)
        merged_data = merged_data.loc[merged_data.Date > "1992-06-01"]
        merged_data_np = merged_data.to_numpy()
        merged_data_np = np.array(merged_data_np[:, 1:], dtype=np.float64)

    elif ((np.sum(Tot_fact_exposure[:7]!=0)+np.sum(Tot_fact_exposure[14:]!=0))==0)|(Tot_fact_exposure.shape[0]==8): #If only US make use of longer time horzion, and add inflation
        US_cpi=pd.read_csv("CPI_US.csv",parse_dates=True)
        US_cpi['DATE'] = pd.to_datetime(US_cpi['DATE'])      
        US_cpi = US_cpi.rename(columns={'DATE': 'Date'})
        US_cpi['CPILFESL']=US_cpi['CPILFESL'].pct_change()
        merged_data= pd.merge(factors_ff6_US,US_cpi, on='Date') #add inflation to the data
        merged_data_np = merged_data.to_numpy()
        merged_data_np = np.array(merged_data_np[:, 1:], dtype=np.float64)
        Tot_fact_exposure = np.append(Tot_fact_exposure[7:14],-1)#add inflation loading
    else:
        merged_data_np = merged_data.to_numpy()
        merged_data_np = np.array(merged_data_np[:, 1:], dtype=np.float64)

    return merged_data_np, Tot_fact_exposure



Net_of_costs=1-Transaction_costs/100 #calculate the net of transaction cost rate


#Now the real core of the simualtion

@jit(nopython=True) #this is too make the code more efficient
def calculate_values(months, Data,Tot_fact_exposure,nr_sim,start_cap, monthly_cap, monthly_correction,Net_of_costs, haircut_min, haircut_max):
    Value = np.zeros((nr_sim,months)) #simulate DCA 
    Value_fixed_inv = np.zeros((nr_sim ,months)) #simulate one fixed investment at begin of the period

    if np.shape(Tot_fact_exposure)[0]==8: #if only US
        nr_markets= 1
    else: #if not only US
        nr_markets=4
    for k in range( nr_sim): #start simulation
        # Create a copy of Tot_fact_exposure for this simulation
        Tot_fact_exposure_sim = Tot_fact_exposure.copy()
        
        lenght = np.random.randint(min_bootstrap, max_bootstrap+1, size=int(months/min_bootstrap)) #draw length of the period
        start = np.empty_like(lenght) 
        for i in range(len(lenght)): 
            start[i] = np.random.randint(0, np.shape(Data)[0]-lenght[i]) #draw a start date in between start date of the sample and end date of the sample-length of the period
        indices = []
        for a, b in zip(lenght, start):
            indices.extend(np.arange(b, b+a)) #add all the draws toghetter to make a hyhtotheical time series
        indices = np.array(indices)[:months] #cut excess length of the bootstrap indices
        for i in range(nr_markets): #apply the haircut here
            Tot_fact_exposure_sim[7*i+1:7*(i+1)-1]=Tot_fact_exposure_sim[7*i+1:7*(i+1)-1]*(1-np.random.uniform(haircut_min, haircut_max,size=(1,5)))
        returns= 1+np.dot(Data[indices,:], Tot_fact_exposure_sim.T)+monthly_correction #calculate the monthly returns over the whole period
        Value[k,0] = start_cap*Net_of_costs  #start investment for DCA analysis
        Value_fixed_inv[k,0]=start_cap*Net_of_costs #start investement for buy once and hold.
        for i in range(1,months): # iterate over the number of months
            Value[k,i]  = Value[k,i-1] *returns[i] +monthly_cap*Net_of_costs
            Value_fixed_inv[k,i]  = Value_fixed_inv[k,i-1] *returns[i] #current value of etf's is last months value of etf's * monthly returns

    return Value, Value_fixed_inv #add the values of the etf's toghetter to get portfolio value per month per simulation



#histogram function
def create_histogram_gif(Returns_A, Returns_B, Start_invest, Monthly_DCA, portfolio_name_A, portfolio_name_B):
    # Create a list to store each frame of the GIF
    images = []
    indic=np.mean(Returns_A)
    if indic>2: #to check if we are working with returns or anualized retruns, very patch way to do this, can go wrong, check later
    # Define the bins for the histogram
        bins = np.linspace(min(np.percentile(Returns_A[:,-1], 0.01),np.percentile(Returns_B[:,-1], 0.01)), max(np.percentile(Returns_A[:,-1], 99),np.percentile(Returns_B[:,-1], 99)), 50)
    else:
        bins = np.linspace(0.85,1.30, 50)


    # Define the range for the y-axis (you can adjust these values as needed)
    y_min = 0

    # Loop over each year (assuming Returns is monthly data)
    for i in range(11, Returns_A.shape[1], 12):
        if indic>2: #to check if we are working with returns or anualized retruns, very patch way to do this, can go wrong, check later
            y_max=max(np.shape(Returns_A)[0]*(1-(2*(i-11)/np.shape(Returns_A)[1])),np.shape(Returns_A)[0]/4)
            title= f'DCA of {Monthly_DCA} after {i//12+1} years, haircut range of [{haircut_min}, {haircut_max}]'
        else:
            y_max=np.shape(Returns_A)[0]/4
            title= f'Annualized returns {i//12+1} years, haircut range of [{haircut_min}, {haircut_max}]'

        fig, ax = plt.subplots(figsize=(10, 6))

        # Create histogram with fixed bins for Returns_A
        ax.hist(Returns_A[:, i], bins=bins, color='blue', alpha=0.5, label=f'{portfolio_name_A}')

        # Create histogram with fixed bins for Returns_B
        ax.hist(Returns_B[:, i], bins=bins, color='red', alpha=0.5, label=f'{portfolio_name_B}')

        ax.set_title(title, fontsize=16)
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

#function to generate tables
def Table(data,months, title_port, table_text):
    percentiles = [1, 10, 25, 50, 75, 90, 99]
    labels = ['Worst 1%', 'Worst 10%', 'Worst 25%', 'Median', 'Best 25%', 'Best 10%', 'Best 1%']
    years = [1, 3, 5, 10, 15, 20,30,40,60,80]

    # Convert years to months
    indice= [year * 12-1 for year in years]
    indice = [i for i in indice if i < months] #select only years within the month range

    # Initialize an empty dataframe to store the results
    df = pd.DataFrame(index=labels)
    # Calculate the percentiles for each year
    for month in indice:
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
  

  


# The main function
def calculate_returns_and_plot(months, merged_data, nr_sim, Start_invest, Monthly_DCA,Net_of_costs,Tot_fact_exposure_A,Tot_fact_exposure_B, Monthly_alfa_min_cost_A,Monthly_alfa_min_cost_B, haircut_min, haircut_max):
    #retrieve relevant data per portfolio
    merged_data_np_A, Tot_fact_exposure_A = process_data(merged_data, factors_ff6_US, Tot_fact_exposure_A )
    merged_data_np_B, Tot_fact_exposure_B = process_data(merged_data, factors_ff6_US, Tot_fact_exposure_B )

    #define data type for JIT function
    Tot_fact_exposure_A=np.array(Tot_fact_exposure_A, dtype=np.float64)
    Tot_fact_exposure_B=np.array(Tot_fact_exposure_B, dtype=np.float64)

    #call the main function 
    Value_A, Value_fixed_inv_A = calculate_values(months, merged_data_np_A, Tot_fact_exposure_A, nr_sim, Start_invest, Monthly_DCA, Monthly_alfa_min_cost_A,Net_of_costs, haircut_min, haircut_max)
    Value_B, Value_fixed_inv_B = calculate_values(months, merged_data_np_B, Tot_fact_exposure_B, nr_sim, Start_invest, Monthly_DCA, Monthly_alfa_min_cost_B,Net_of_costs, haircut_min, haircut_max)


    #calcualted the anualized rate of returns per month for both portfolios. 
    IRR_A=np.power(Value_fixed_inv_A/Start_invest,12/np.arange(1,months+1))
    IRR_B=np.power(Value_fixed_inv_B/Start_invest,12/np.arange(1,months+1))

    #make the portfolio names for the grpahs 
    title_port_A =  "Portfolio A"
    title_port_B =  "Portfolio B"

    #calcualte the invested amount
    invested = Start_invest + Monthly_DCA * np.arange(months)

    #calcualted the percentiles for the graphs
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
    table_text=f'End value with DCA of {Monthly_DCA}'
    Table(Value_A, months,title_port_A, table_text)
    Table(Value_B,months, title_port_B, table_text)
    table_text= 'Anualized returns'
    Table(IRR_A, months,title_port_A, table_text)
    Table(IRR_B,months, title_port_B, table_text)

# Call the function
calculate_returns_and_plot(months, merged_data, nr_sim, Start_invest, Monthly_DCA,Net_of_costs,Tot_fact_exposure_A,Tot_fact_exposure_B, Monthly_alfa_min_cost_A,Monthly_alfa_min_cost_B, haircut_min, haircut_max)





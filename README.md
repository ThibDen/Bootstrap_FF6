# Bootstrap_FF6
Bootstrap_portfolio_simulators_V0 : Bootstrap of you're portfolio based on a 6 factor model. 
Very rough first version. Some comments in the code to make it understable. Probably still some bugs when you add different markets so be carefull with results. 
Hope to rewrite it a bit better in the future, all suggestions are welcome. 
Inflation correction only works for when you are considering US only portfolios. If you also add non-US market, inflation is ignored. 

The code Bootstrap_market_loadings_simulators_V0 gives the same functionality but based on factor loadings per market instead of ETF specific. It thus assumes frictionless rebalancing (keeping exposure constant usually require rebalancing which create transaction costs/taxes). Inflation correction only works for when you are considering US only portfolios. If you also add non-US market, inflation is ignored. 


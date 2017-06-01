## @knitr 1a
library (quantmod)
getSymbols("AMZN",from="2003-01-02", to="2015-04-30")
adjclosed=diff(log(as.numeric(AMZN$AMZN.Adjusted)))
source("RMfit.R")
negadjclosed=-adjclosed
m1=RMfit(negadjclosed)
sqrt(10)*0.06178949

## @knitr 1b
library(fGarch)
m2=garchFit(~garch(1,1),data=negadjclosed,trace=F)
m2@fit$matcoef
predict(m2,1)
source("RMeasure.R")
RMeasure(-0.001299307,0.0316129)

## @knitr 1c
m3=garchFit(~garch(1,1),data=negadjclosed,trace=F,cond.dist="std")
m3@fit$matcoef
predict(m3,1)
RMeasure(-0.0006388537,0.02465634)

## @knitr 2
require(evir)
source("evtVaR.R")
m4=gev(negadjclosed,block=21)
m4$par.ests
m4$par.ses
evtVaR(m4$par.ests[1],m4$par.ests[2],m4$par.ests[3])
0.06381642*10^(0.31427501)

## @knitr 3
m5=gpd(negadjclosed,0.035)
riskmeasures(m5,c(0.99))
m6=gpd(negadjclosed,0.045)
riskmeasures(m6,c(0.99))

## @knitr 4
getSymbols("KO", from="2003-01-02", to="2015-04-30")
koadj=diff(log(as.numeric(KO$KO.Adjusted)))
negkoadj=-koadj
cor(adjclosed,koadj)
m7=RMfit(negkoadj)
sqrt(61789.5^2+18065.8^2+2*61789.5*18065.8*0.304)
x1=negadjclosed+negkoadj
x2=negadjclosed-negkoadj
m8=RMfit(x1)
m9=RMfit(x2)
c1=(m8$volatility^2-m9$volatility^2)*.25
rho=c1/(m1$volatility*m7$volatility)
rho[length(rho)]
sqrt(61789.5^2+18065.8^2+2*61789.5*18065.8*0.02680454)


## @knitr 5
RMfit(adjclosed)
sqrt(61789.49^2+18065.8^2-2*61789.49*18065.8*0.304)
m10=garchFit(~garch(1,1),data=negkoadj, trace=F, cond.dist='std')
m10@fit$matcoef
predict(m10,1)
RMeasure(predict(m10,1)$meanForecast,predict(m10,1)$meanError, cond.dist='std', df=5.207)
RMeasure(predict(m3,1)$meanForecast,predict(m3,1)$meanError, cond.dist='std', df=3.689)
sqrt(64888.39^2+19976.16^2+2*19976.16*64888.39*.304)

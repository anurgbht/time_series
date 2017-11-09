rm(list=ls())

library(glm)
library(ggplot2)

setwd('D:/OneDrive - Tata Insights and Quants, A division of Tata Industries/Confidential/Projects/Steel/Strategic Procurement/data/Raw Materials/')

dat = read.csv('temp_regression.csv')
dat = na.omit(dat)
mod_dat = dat[,-c(1)]
mod_dat = mod_dat[,c(1,10,15,26,51)]
#mod_dat = mod_dat[,c(1,2,10,15,19,26,27,28,38,52)]
platts = dat$platts
plot(platts,type = 'l',panel.first = grid())

model = lm(platts ~ ., data = mod_dat)

summary(model)

tt = predict(model,mod_dat)
dates = dat$date
plot(platts,type='l',col='red',lwd=2,panel.first = grid())
lines(tt,col='blue',lwd=2)
legend("bottomright",c('predicted','actual'),lty=c(1,1),lwd=c(2.5,2.5),col=c("blue","red"))

plot(mod_dat$cny_per_usd_monthly_BidClose,col='green',type = 'l',lwd=2,panel.first = grid())

MAPE(platts,tt)
MSE(platts,tt)

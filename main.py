#from datetime import datetime as dt
import numpy as np
import pandas as pd
from cliodynamics import Cliodynamic
import dataset


t = dataset.tuition()
w = dataset.relativeWage() 
l = dataset.lifeExpectancy()
f = dataset.foreignBornModel()
s = dataset.stature()
m = dataset.marriageAge()
i = dataset.inequality() 
p = dataset.polarization()

minimums = [round(t._min_xx),round(w._min_xx),round(l._min_xx),round(f._min_xx),round(s._min_xx),round(m._min_xx),round(i._min_xx),round(p._min_xx)]
maximums = [round(t._max_xx),round(w._max_xx),round(l._max_xx),round(f._max_xx),round(s._max_xx),round(m._max_xx),round(i._max_xx),round(p._max_xx)]

minimum_year = max(set(minimums), key=minimums.count)
#maximum_year = max(set(maximums), key=maximums.count)
maximum_year = max(maximums)

years = np.array([minimum_year])
values = np.array([(f.eval(minimum_year) + s.eval(minimum_year) + m.eval(minimum_year) + l.eval(minimum_year) + w.eval(minimum_year) + t.eval(minimum_year) + i.eval(minimum_year) + p.eval(minimum_year)) / 8])

for x in range(minimum_year + 1,maximum_year + 1):
    avgNum = (f.eval(x) + s.eval(x) + m.eval(x) + l.eval(x) + w.eval(x) + t.eval(x) + i.eval(x) + p.eval(x)) / 8
    years = np.append(years,x)
    values = np.append(values,avgNum)

df = pd.DataFrame(data={'Year':years, 'Well-Being Index':values})

# Set indices to datetime
df = df.set_index('Year')
df.index = pd.to_datetime(df.index, format='%Y')

c = Cliodynamic('Well-Being Index', df, normalize=False)

for x in range(1650,2100):
    avgNum = (f.eval(x) + s.eval(x) + m.eval(x) + l.eval(x) + w.eval(x) + t.eval(x) + i.eval(x) + p.eval(x)) / 8
    fitSin = c._fitFunc(x)
    print(str(x) + "," + str(f.eval(x)) + "," + str(w.eval(x)) + "," + str(s.eval(x)) + "," + str(l.eval(x)) + "," + str(m.eval(x)) + "," + str(t.eval(x)) + "," + str(i.eval(x)) + "," + str(p.eval(x)) + "," + str(avgNum) + "," + str(fitSin))

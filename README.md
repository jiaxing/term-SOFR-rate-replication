# term-sofr-replication
Computing the Term SOFR rate implied by SOFR futures prices. 
This is the supporting code of the article in https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4566882.

The project includes 3 files:
- cmeProjectSupportingFunctions: set of basic functions to support the code;
- fromOHLCtoOFFICIALprices: script for converting futures prices from Open High Low Close to a single official price for the day, to be used to calibrate the curve. The script is consistent with CME Group's methodology, but it can be easily converted to compute the VWAP of any hedging strategy;
- TermSOFRRateEstimation: main script to calibrate the curve using some given official prices for the day and computing the Term SOFR rates.


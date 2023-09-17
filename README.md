**Computing the Term SOFR rate implied by SOFR futures prices**.  
This is the supporting code of the article in https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4566882.

The project includes 3 files:
- _cmeProjectSupportingFunctions_: set of basic functions to support the code;
- _fromOHLCtoOFFICIALprices_: script for converting futures prices from Open High Low Close to a single official price for the day, to be used to calibrate the curve. The script is consistent with CME Group's methodology, but it can be easily converted to compute the VWAP of any hedging strategy;
- _TermSOFRRateEstimation_: main script to calibrate the curve and computing the Term SOFR rates.


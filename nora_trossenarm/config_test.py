"""
BackGroud: 
Trossen arm 1.9.x python package containes cartissean coord conversion. 
But we can't upgrade arm's driver firmware to be compatible. 

This script aim to test driver's firmware version. 

Further investigation is needed to see if we can upgrade firmware. 
"""

import trossen_arm

d = trossen_arm.TrossenArmDriver()
d.configure(trossen_arm.Model.wxai_v0, trossen_arm.StandardEndEffector.wxai_v0_base, '192.168.2.2', False)
print('Controller version:', d.get_controller_version())
print('Driver version:', d.get_driver_version())
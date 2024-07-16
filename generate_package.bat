@echo off

Rem Generate a package that can be copied into Tensorflow projects

MD .\package\parameters
XCOPY .\dev_src\parameters\lpips_tf .\package\parameters\lpips_tf /E

MD .\package\loss_fns
COPY .\dev_src\loss_fns\lpips_base_tf.py .\package\loss_fns

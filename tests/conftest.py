import os

# Prevent fortran from buffering to silence multinest output during the tests
# Must be set before pymultinest is imported
os.environ.setdefault("GFORTRAN_UNBUFFERED_ALL", "1")

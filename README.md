## To Run This

- Git clone the repo: `git clone https://github.com/NotCleo/CSE311-NeuralNetworkBasedCompression.git`
- Create a virtual environment (if required): `python3 -m venv venv`, `source venv/bin/activate`
- Install all the requirements: `pip install -r requirements.txt`
- Run the scripts: `python3 main.py`



All the training was carried out on the following machine on Colab (Set Runtime Type : T4): 

Architecture:                x86_64
  CPU op-mode(s):            32-bit, 64-bit
  Address sizes:             46 bits physical, 48 bits virtual
  Byte Order:                Little Endian
CPU(s):                      2
  On-line CPU(s) list:       0,1
Vendor ID:                   GenuineIntel
  Model name:                Intel(R) Xeon(R) CPU @ 2.20GHz
    CPU family:              6
    Model:                   79
    Thread(s) per core:      2
    Core(s) per socket:      1
    Socket(s):               1
    Stepping:                0
    BogoMIPS:                4399.99
    Flags:                   fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pg
                             e mca cmov pat pse36 clflush mmx fxsr sse sse2 ss h
                             t syscall nx pdpe1gb rdtscp lm constant_tsc rep_goo
                             d nopl xtopology nonstop_tsc cpuid tsc_known_freq p
                             ni pclmulqdq ssse3 fma cx16 pcid sse4_1 sse4_2 x2ap
                             ic movbe popcnt aes xsave avx f16c rdrand hyperviso
                             r lahf_lm abm 3dnowprefetch ssbd ibrs ibpb stibp fs
                             gsbase tsc_adjust bmi1 hle avx2 smep bmi2 erms invp
                             cid rtm rdseed adx smap xsaveopt arat md_clear arch
                             _capabilities
Virtualization features:     
  Hypervisor vendor:         KVM
  Virtualization type:       full
Caches (sum of all):         
  L1d:                       32 KiB (1 instance)
  L1i:                       32 KiB (1 instance)
  L2:                        256 KiB (1 instance)
  L3:                        55 MiB (1 instance)
NUMA:                        
  NUMA node(s):              1
  NUMA node0 CPU(s):         0,1
Vulnerabilities:             
  Gather data sampling:      Not affected
  Indirect target selection: Vulnerable
  Itlb multihit:             Not affected
  L1tf:                      Mitigation; PTE Inversion
  Mds:                       Vulnerable; SMT Host state unknown
  Meltdown:                  Vulnerable
  Mmio stale data:           Vulnerable
  Reg file data sampling:    Not affected
  Retbleed:                  Vulnerable
  Spec rstack overflow:      Not affected
  Spec store bypass:         Vulnerable
  Spectre v1:                Vulnerable: __user pointer sanitization and usercop
                             y barriers only; no swapgs barriers
  Spectre v2:                Vulnerable; IBPB: disabled; STIBP: disabled; PBRSB-
                             eIBRS: Not affected; BHI: Vulnerable
  Srbds:                     Not affected
  Tsa:                       Not affected
  Tsx async abort:           Vulnerable

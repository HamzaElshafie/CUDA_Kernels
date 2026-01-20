# Fundamental GPU Performance Q & A (CUDA + ML Infra)

Simple, accurate explanations of core performance concepts that commonly appear in CUDA and ML infrastructure interviews.


## 1) What is throughput, and how is it different from latency?

**Answer**  
Throughput is how much work you finish per unit time, for example tokens per second, samples per second, or FLOPs per second.  
Latency is how long one unit of work takes, for example time for one inference request or one kernel launch.

GPUs are usually optimised for throughput. Many real world performance problems come from optimising throughput when latency matters, or the opposite.


## 2) What do people mean by “compute bound” versus “memory bound”?

**Answer**  
Compute bound means performance is limited by arithmetic throughput, the math pipelines are the bottleneck.  
Memory bound means performance is limited by how fast data can be moved from memory to the compute units.

A simple test:  
If you add more math and runtime increases, it was compute bound.  
If you add more math and runtime barely changes, it was memory bound.


## 3) What is memory bandwidth in simple terms?

**Answer**  
Memory bandwidth is how many bytes per second the GPU can deliver from main memory.

Peak bandwidth is a theoretical upper bound. Achieved bandwidth depends on access patterns, alignment, coalescing, and how many memory requests are in flight.


## 4) What is compute throughput?

**Answer**  
Compute throughput is how many arithmetic operations the GPU can execute per second. It depends on data type and hardware units, such as FP32 cores or tensor cores for FP16 and BF16.

Peak compute throughput is also an upper bound. Real kernels usually achieve less due to stalls, instruction mix, and memory limits.


## 5) What is arithmetic intensity?

**Answer**  
Arithmetic intensity is the number of floating point operations performed per byte moved from main memory.

Formula:  
Arithmetic intensity = FLOPs / bytes moved

Low intensity means memory bound. High intensity means compute heavy.


## 6) What is the roofline model and why is it useful?

**Answer**  
The roofline model says achievable performance is limited by either compute or memory bandwidth.

Achievable FLOPs per second ≤ min(peak compute, peak bandwidth × arithmetic intensity)

It helps decide where to optimise:
- If memory bound, reduce bytes or improve memory efficiency
- If compute bound, improve math pipeline usage or tensor core utilisation


## 7) What is a roofline plot?

**Answer**  
A roofline plot shows performance versus arithmetic intensity.

- X axis: arithmetic intensity  
- Y axis: performance

There is a sloped line for the memory bandwidth limit and a flat line for the compute limit. Kernels sit below these ceilings.


## 8) What does “speed of light” mean in performance discussions?

**Answer**  
It refers to the fact that signal propagation has physical limits. Even at a fraction of the speed of light, moving data across the GPU and memory takes time.

Some latency cannot be removed, only hidden with parallelism.


## 9) Memory latency versus memory bandwidth, what is the difference?

**Answer**  
Latency is how long a single memory request takes.  
Bandwidth is how much data can be transferred per second with many requests.

A kernel can be latency bound without using much bandwidth if it cannot issue enough concurrent requests.


## 10) What does latency hiding mean on a GPU?

**Answer**  
When one warp waits for memory, the GPU schedules another ready warp. With enough warps and independent work, memory latency is hidden.

This is why occupancy can matter, but only when latency is the bottleneck.


## 11) What is occupancy, and why is it misunderstood?

**Answer**  
Occupancy is the fraction of the maximum number of warps that can reside on an SM.

It is misunderstood because:
- High occupancy does not guarantee high performance
- Many kernels perform best at moderate occupancy
- Occupancy is a tool, not a goal


## 12) What is achieved occupancy versus theoretical occupancy?

**Answer**  
Theoretical occupancy is calculated from registers, shared memory, and block size.  
Achieved occupancy is what actually happens during execution.

Both matter, but achieved occupancy reflects reality.


## 13) What is instruction level parallelism (ILP)?

**Answer**  
ILP means having multiple independent instructions per thread that can execute in parallel.

High ILP can hide latency even when occupancy is low.


## 14) What is memory coalescing?

**Answer**  
Memory coalescing happens when threads in a warp access nearby addresses, allowing the hardware to combine requests into fewer memory transactions.

Poor coalescing wastes bandwidth.


## 15) What is memory efficiency?

**Answer**  
Memory efficiency measures how much of the transferred data is actually useful.

Low efficiency means extra bytes are moved due to poor alignment or access patterns.


## 16) Why can a memory bound kernel reach only 50% of peak bandwidth?

**Answer**
Reasons include:
- Misalignment or partial coalescing
- Not enough concurrent memory requests
- Instruction overhead limiting issue rate
- Cache level bottlenecks

Peak bandwidth assumes ideal conditions.


## 17) What is SM utilisation versus compute utilisation?

**Answer**  
SM utilisation measures how busy the SM is overall.  
Compute utilisation measures how busy the math pipelines are.

High SM utilisation does not guarantee high compute utilisation.


## 18) What is a stall?

**Answer**  
A stall occurs when a warp cannot issue its next instruction.

Common stall types:
- Memory dependency
- Instruction dependency
- Barrier synchronisation
- Pipeline unavailability


## 19) What is register pressure?

**Answer**  
Register pressure refers to how many registers each thread uses.

High register usage can reduce occupancy and cause register spills, which are very costly.


## 20) What is local memory?

**Answer**  
Local memory is per thread storage that spills to device memory when registers are insufficient.

Despite the name, it is slow and usually signals a performance problem.


## 21) What is shared memory and when does it help?

**Answer**  
Shared memory is on chip memory shared by threads in a block.

It helps when data is reused multiple times within a block. Without reuse, it adds overhead.


## 22) What are shared memory bank conflicts?

**Answer**  
Shared memory is divided into banks. When multiple threads access different addresses in the same bank, accesses serialize and slow down.

Padding or indexing changes can fix this.


## 23) What is kernel launch overhead?

**Answer**  
Launching a kernel has a fixed CPU and driver cost.

It matters most for very small or short running kernels. Fusion and CUDA Graphs reduce this overhead.


## 24) What is kernel fusion and why does it matter?

**Answer**  
Kernel fusion combines multiple operations into one kernel.

Benefits:
- Fewer kernel launches
- Fewer global memory reads and writes
- Better locality using registers

## 25) Can fusion hurt performance?

**Answer**  
Yes. Fusion can increase register usage, reduce occupancy, and cause spills.

The fix is to fuse carefully and check for spills and end to end runtime.


## 27) What is effective bandwidth?

**Answer**  
Effective bandwidth = useful bytes moved / runtime

It shows how close you are to hardware limits, assuming useful bytes are defined correctly.

---

## 28) Why can FLOP count be misleading?

**Answer**  
Because many kernels are not compute bound.

Also:
- Not all operations cost the same
- Memory and instruction overhead can dominate


## 29) Arithmetic intensity versus operational intensity?

**Answer**  
They usually mean the same thing: operations per byte.

Arithmetic intensity is the common roofline term.


## 30) Example arithmetic intensity comparison

**Answer**
Elementwise add:
- Very low arithmetic intensity
- Usually memory bound

Matrix multiply:
- High arithmetic intensity
- Often compute bound

## 31) What does it mean to be roofline limited?

**Answer**  
It means the kernel is close to the theoretical performance ceiling for its arithmetic intensity.

Further micro optimisation yields limited gains. Bigger changes are needed.


## 32) One performance workflow to remember

**Answer**
1. Measure end to end performance
2. Identify the critical path
3. Classify hot kernels using roofline thinking
4. Use profiler metrics to confirm
5. Apply the correct optimisation lever
6. Validate end to end improvement

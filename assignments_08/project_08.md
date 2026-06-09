# Project 08 -- Intro and Cost Analysis

## Video Link for azure 
https://youtu.be/q7wB3KmeUXw

---
## Cost analysis video link
https://youtu.be/5RT4qGqLrRE

## Cost Summary

I used the Azure Pricing Calculator (East US, pay as you go) to estimate two scenarios.

Scenario A -- Lightweight Compute: A Standard_B1s VM (1 vCPU, 1 GB RAM) running about 160 hours a month costs $2.24 per month, $26.88 a year.

Scenario B -- Heavy Analytics Workload: This one has three parts:

- GPU VM (Standard_NC6s_v3) running 24/7: $2,233.80/month
- Azure SQL Database (General Purpose, 4 vCores): $737.12/month
- Blob Storage with 1 TB of data: $22.34/month

Together that is about $2,993 per month, or almost $36,000 a year

### What I found surprising

- The GPU VM is by far the most expensive part. It costs more per hour ($3.06) than the small VM costs for a whole day.
- Storing 1 TB of data is really cheap ($22/month) compared to the compute. I expected storage to cost more.
- The SQL database was more expensive than I thought ($737/month), and a big chunk of that is just the SQL license.
- Hours matter a lot. The small VM is cheap partly because it only runs 8 hours a day on weekdays. Leaving the GPU VM on 24/7 is what makes it so expensive, so shutting VMs down when you are not using them would save a lot of money.



# Run this in Azure Cloud Shell after completing the Cost Analysis above.

# Fill in the hourly rates from your two Pricing Calculator estimates.
rate_a = 0.014  # Standard_B1s hourly rate (Scenario A)
rate_b = 3.060  # Standard_NC6s_v3 hourly rate (Scenario B, VM only)

hours_a = 160   # Scenario A: 8h/day, 5 days/week, ~4 weeks
hours_b = 730   # Scenario B: always on

cost_a = rate_a * hours_a
cost_b = rate_b * hours_b

print("=== Monthly Cost Estimates ===")
print(f"Scenario A (lightweight):       ${cost_a:.2f}")
print(f"Scenario B (GPU VM only):       ${cost_b:.2f}")

if cost_a > 0:
    print(f"Scenario B VM costs {cost_b / cost_a:.1f}x more than Scenario A")


### Script output

I ran `project_08.py` in Azure Cloud Shell and it printed:

=== Monthly Cost Estimates ===
Scenario A (lightweight):       $2.24
Scenario B (GPU VM only):       $2233.80
Scenario B VM costs 997.2x more than Scenario A

The calculated costs matched the Pricing Calculator exactly: $2.24/month for the B1s and $2,233.80/month for the NC6s v3 VM. That makes sense because the calculator is doing the same math (hourly rate x hours). The script only covers the VMs, so it does not include the SQL database or storage from Scenario B.

---
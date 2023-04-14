# using Pkg
#Pkg.add(["DifferentialEquations", "Plots", "SciMLSensitivity", "OrdinaryDiffEq","Zygote", "Optimization", "OptimizationNLopt", "OptimizationFlux","Flux", "ProgressMeter", "MAT"])
using Pkg; Pkg.activate("optim")

using Random
using DifferentialEquations
using Plots
using SciMLSensitivity
using OrdinaryDiffEq
using Zygote
using Optimization
using OptimizationNLopt
using OptimizationFlux
using Flux
using ProgressMeter
using MAT
using Base.Threads
using Serialization

# Replace "your_file.mat" with the path to your .mat file
filename = "virtualpatients_TMZ+GDC_article.mat"

# Load the .mat file
matfile = matopen(filename)
data = read(matfile, "A")
close(matfile)

function pk_pd!(du, u, p, t)
    # Unpack parameters
    gamma_1, psi, C0, D0, r, K, BW, IC50_1, Imax_1, IC50_2, gamma_2, Imax_2, xi, VD1, Cl1, k23, ka1, k32, ka2, V2, kel, k12, k21 = p[1:length(ode_params)]
    
    # Unpack state variables
    C, D, AbsTMZ, PlaTMZ, CSFTMZ, AbsGDC, PlaGDC, PeriphGDC = u

    cCSFTMZ = CSFTMZ / 140
    cPlaGDC = PlaGDC / 1000 * V2

    exp1 = real(complex(cCSFTMZ)^gamma_1/(psi*IC50_1)^gamma_1)
    exp2 = real(complex(xi*cPlaGDC)^gamma_2/(psi*IC50_1)^gamma_2)
    E = (Imax_1*exp1+Imax_2*exp2+(Imax_1+Imax_2-Imax_1*Imax_2)*exp1*exp2)/(exp1+exp2+exp1*exp2+1) 

    t1=-1/r*log(complex(log(complex(C/K))/log(complex(C0/K))))
    t2=t1+3*24
    fun = K*exp(log(complex(17.7/K))*exp(-r*t2))
    delta=real(E/72*(fun/C))

    dC = real(abs((C*r*log(complex(K/C))))-max(0,delta*C))
    dD = real(max(0,delta*C))

    dAbsTMZ = -ka1 * AbsTMZ
    dPlaTMZ = ka1 * AbsTMZ - Cl1 / VD1 * PlaTMZ - k23 * PlaTMZ + k32 * CSFTMZ
    dCSFTMZ = k23 * PlaTMZ - k32 * CSFTMZ

    El = kel * PlaGDC
    dAbsGDC = -ka2 * AbsGDC
    dPlaGDC = ka2 * AbsGDC - El + k21 * PeriphGDC - k12 * PlaGDC
    dPeriphGDC = -k21 * PeriphGDC + k12 * PlaGDC

    du .= [dC, dD, dAbsTMZ, dPlaTMZ, dCSFTMZ, dAbsGDC, dPlaGDC, dPeriphGDC]
end

hours = 24
end_time = 28*5
end_treat = 42

u0 = [17.7,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
tspan = (0,end_time-1).*hours
_gamma_1, _psi, _C0, _D0, _r         , _K    , _BW, _IC50_1               , _Imax_1, _IC50_2  , _gamma_2, _Imax_2, _xi                               , _VD1, _Cl1, _k23     , _ka1, _k32, _ka2 , _V2, _kel, _k12, _k21 = 0.5689, 1   ,17.7,   0, 0.007545/24, 158.04, 70 , 5.807*0.000001*194.151, 2.905  , 0.0005364, 0.26    , 0.16   , (5.807*0.000001*194.151)/0.0005364, 30.3, 5   , 7.2*10^-3, 5.8 , 0.76, 0.234, 77 , 17.4, 67.2, 171.6
ode_params = [_gamma_1,_psi,_C0,_D0,_r,_K,_BW,_IC50_1,_Imax_1,_IC50_2,_gamma_2,_Imax_2,_xi,_VD1,_Cl1,_k23,_ka1,_k32,_ka2,_V2,_kel,_k12,_k21];

function generate_z(x, y)
    z = Array{Float64, 2}(undef, 23, 400)
    x_replace_indices = [15, 16, 17, 19, 20, 21]
    z .= x
    z[x_replace_indices, :] .= y
    return z
end
patients = transpose(generate_z(ode_params, transpose(data[:,[2,3,4,6,7,8]])));

function relu(x)
    max(0,x)
end

function spaced_list(p, n, m, b=1)
    # Initialize an empty list
    spaced_list = []
    # Initialize a counter variable
    counter = b
    # Use a while loop to iterate through the range of integers
    while counter <= p
        # Append `n` integers spaced by 1 to the list
        for i in 1:n
            spaced_list = [spaced_list; counter]
            counter += 1
            # Check if the counter has reached the end of the range
            if counter > p
                break
            end
        end
        # Add `m` to the counter to create the jump
        counter += m
    end
    return spaced_list
end

hours = 24
end_time = 28*5
end_treat = 42

tmz_treat_dosetimes = spaced_list(end_treat,1,0,0).*hours
tmz_adjuv_dosetimes = spaced_list(end_time,5,23,end_treat+28).*hours
gdc_dosetimes = spaced_list(end_time-1,21,7,0).*hours

inject_times = sort(unique([gdc_dosetimes;tmz_treat_dosetimes;tmz_adjuv_dosetimes]));

rng = Random.default_rng()
Random.seed!(rng, 42)

tmz_treat_dose = 0.075*1.7
tmz_adjuv_dose = 0.150*1.7
# gdc_dose = rand(length(gdc_dosetimes)).*0.4;

# p=[ode_params;gdc_dose];
function affect_dose!(integrator)
    #should make it depend on the time maybe P as dictionary
    ev = 0
    if integrator.t in tmz_treat_dosetimes
        integrator.u[3] += tmz_treat_dose
        ev += 1
    else 
        nothing
    end
    if integrator.t in tmz_adjuv_dosetimes
        integrator.u[3] += tmz_adjuv_dose
        ev += 1
    else 
        nothing
    end
    if integrator.t in gdc_dosetimes
        hit_gdc = integrator.p[length(ode_params)+1:end][findall(x->x==integrator.t,gdc_dosetimes)][1]
        integrator.u[6] += relu(hit_gdc)
        ev += 1
    else 
        nothing
    end
    if ev == 0
        nothing#println("this should not get here!")
    else 
        nothing
    end
end

hit = PresetTimeCallback(inject_times, affect_dose!);

u0 = [17.7,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
tspan = (0,end_time+7).*hours
p=[ode_params;rand(length(gdc_dosetimes)).*0.4]

prob = ODEProblem(pk_pd!,u0,tspan,p)
sol = solve(prob, tstops=inject_times, callback=hit);

function loss(p)
    sol = solve(prob, tstops=inject_times, callback=hit, p=p, saveat=0.1,
    sensealg=ForwardDiffSensitivity(convert_tspan=true))
    #sensealg=InterpolatingAdjoint(checkpointing=true))
    #sensealg=ReverseDiffVJP())
    #sensealg=BacksolveAdjoint())

    sols = Array(sol)
    c = sols[1,:]
    min_cell =  2.8727 #19269.3988 # last:  2.8727
    max_cell = 30.3599 #67600.655  # last: 30.3599
    final_cell = c[end]
    #cell_auc = (sum((c[1:end-1] + c[2:end]) .* 0.1) / 2)
    cell_normed = (final_cell - min_cell)/(max_cell - min_cell)#11.161748645141492

    d2 = sols[7,:]
    min_drug = 0.0
    max_drug = 17.9756 #1.3721
    drug_normed = ((sum((d2[1:end-1] + d2[2:end]) .* 0.1) / 2) - min_drug) / (max_drug - min_drug) #0.005975799220806873
    
    # λ=0.4
    # ncw = (λ)*cell_normed
    # ndw = (1-λ)*drug_normed
    # # Construct the final loss function with an additional term for the inverse relationship
    # inverse_relationship_term = -λ * (1 - λ) * ncw * ndw
    # loss = ncw + ndw + inverse_relationship_term
    α=0.9
    loss = α*cell_normed + (1-α)*drug_normed
    return loss
end

function train(p,optimizer=ADAMW(0.05),epoch=100)
    for _ in 1:epoch
        dose_grads = gradient(loss, p)[1][24:end]
        grads = [zeros(23);dose_grads]
        Flux.Optimise.update!(optimizer, p, grads)
    end
    return p
end


function save_optima(optima, filename)
    open(filename, "w") do file
        serialize(file, optima)
    end
    println("Saved optima to $filename")
end

optimizer = ADAMW(0.05)
optima = zeros(400, 128)
save_filename = "optima_data_3.jls"

Threads.@threads for i in 1:12
    println("Patient: $i")
    vp = patients[i, :]
    p = [vp; rand(length(gdc_dosetimes)) .* 0.1]
    optima[i, :] = train(p, optimizer, 1)
end

# Save final optima after the loop is done
save_optima(optima, save_filename)
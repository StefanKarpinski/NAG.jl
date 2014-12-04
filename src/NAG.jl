module NAG
export
    nag_licence_query, nag_license_query,
    nag_complex_polygamma,
    nag_opt_read!,
    nag_opt_nlp!,
    nag_1d_quad_inf_1,
    nag_zero_cont_func_brent,
    nag_opt_lp!,
    last_nag_error

nag_licence_query() = ccall((:a00acc, :libnagc_nag), Cint, ()) == 1
const nag_license_query = nag_licence_query # US vs UK spelling
nag_license_query() || warn("Cannot acquire a NAG license.")

const NagInt = Int32
const NagComplex = Complex128

type NagError <: Exception
    code::Int
    name::ASCIIString
    message::UTF8String
end

Base.showerror(io::IO, e::NagError) =
    print(io, "NAG function \"$(e.name)\" [$(e.code)] – $(e.message)")

const NAG_ERROR = zeros(Uint8,544)
const nag_errors = Array(NagError)
nag_errors[] = NagError(0, "", "NO_ERROR")

cstr_to_array(p::Ptr{Uint8}, own::Bool = false) =
    pointer_to_array(p, int(ccall(:strlen, Csize_t, (Ptr{Uint8},), p)), own)

function error_handler(msg::Ptr{Uint8}, code::Cint, name::Ptr{Uint8})
    code == 0 && return
    msg = UTF8String(copy(cstr_to_array(msg)))
    name = ASCIIString(copy(cstr_to_array(name)))
    nag_errors[] = NagError(int(code), name, msg)
    throw(nag_errors[])
end
const ptr_error_handler = cfunction(error_handler, Void, (Ptr{Uint8}, Cint, Ptr{Uint8}))

function reset_nag_error()
    fill!(NAG_ERROR, 0)
    unsafe_store!(
        convert(Ptr{Ptr{Void}}, pointer(NAG_ERROR)) + 2*sizeof(Cint) + 512,
        ptr_error_handler
    )
end
reset_nag_error()

last_nag_error() = nag_errors[]

const objfunref = Array(Function)
const confunref = Array(Function)

function objfun_wrapper(
    n::NagInt, x::Ptr{Float64}, objf::Ptr{Float64}, g::Ptr{Float64}, comm::Ptr{NagInt}
)
    x = pointer_to_array(x,int(n),false)
    g = pointer_to_array(g,int(n),false)
    f = unsafe_load(comm)
    unsafe_store!(objf,objfunref[1](x,g,f))
    return nothing
end
const c_objfun = cfunction(objfun_wrapper, Void,
    (NagInt, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{NagInt}))

function confun_wrapper(
    n::NagInt, ncnlin::NagInt, needc::Ptr{NagInt}, x::Ptr{Float64},
    conf::Ptr{Float64}, conjac::Ptr{Float64}, comm::Ptr{NagInt}
)
    needc  = pointer_to_array(needc,int(ncnlin),false)
    x      = pointer_to_array(x,int(n),false)
    conf   = pointer_to_array(conf,int(ncnlin),false)
    conjac = pointer_to_array(conjac,(int(n),int(ncnlin)),false)
    flag   = unsafe_load(comm)
    confunref[1](needc,x,conf,conjac,flag)
    return nothing
end
const c_confun = cfunction(confun_wrapper, Void,
    (NagInt, NagInt, Ptr{NagInt}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{NagInt}))

function nag_complex_polygamma(z::Number, k::Integer)
    ccall((:s14afc, :libnagc_nag), NagComplex,
          (NagComplex, NagInt, Ptr{Void}),
          z, k, NAG_ERROR)
end

# if Base.polygamma doesn't support complex args, use NAG
if !applicable(Base.polygamma, 1, 1+2im)
    Base.polygamma(k::Int, z::Complex) = nag_complex_polygamma(z, k)
    Base.digamma(z::Complex) = polygamma(0, z)
    Base.trigamma(z::Complex) = polygamma(1, z)
end

function nag_opt_read!(name::ByteString, optfile::ByteString, print::Bool = false)
    options = zeros(Uint8,1440)
    ccall((:e04xxc, :libnagc_nag), Void, (Ptr{Void},), options)
    ccall((:e04xyc, :libnagc_nag), Void,
          (Ptr{Uint8}, Ptr{Uint8}, Ptr{Void}, Cint, Ptr{Uint8}, Ptr{Void}),
          name, optfile, options, print, "stdout", NAG_ERROR)
    return options
end

function nag_opt_lp!(
    A         :: Matrix{Float64},
    bl        :: Vector{Float64},
    bu        :: Vector{Float64},
    c         :: Vector{Float64},
    x         :: Vector{Float64};
    optfile   :: ByteString = "",
    transpose :: Bool = true,
)
    # since NAG is row-major
    #   when transpose == true the rows are linear constraints
    #   when transpose == false the columns are linear constraints
    transpose && (A = A')
    
    n = length(c)
    tda, nclin = size(A)

    nclin > 0 && tda >= n   ||
        error("bad linear constraint dimensions (tda ≱ n): $tda ≱ $n")
    length(bl) == n + nclin ||
        error("wrong number of lower bounds: $(length(bl)) ≠ $(n + nclin)")
    length(bu) == n + nclin ||
        error("wrong number of upper bounds: $(length(bu)) ≠ $(n + nclin)")
    length(c) == n          ||
        error("wrong number of coefficients: $(length(c)) ≠ $n")
    length(x) == n          ||
        error("wrong data vector size: $(length(x)) ≠ $n")

    objf = zeros()
    options = isempty(optfile) ? C_NULL :
        convert(Ptr{Void}, nag_opt_read!("e04mfc", optfile))
    comm = zeros(Uint8, 8)

    reset_nag_error()
    ccall((:e04mfc,:libnagc_nag), Void,
           (NagInt, NagInt, Ptr{Float64}, NagInt, Ptr{Float64}, Ptr{Float64},
            Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Void}, Ptr{Void}, Ptr{Void}),
           n, nclin, A, tda, bl, bu, c, x, objf, options, comm, NAG_ERROR)
    return x, objf
end

function nag_opt_nlp!(
    A  :: Matrix{Float64},
    bl :: Vector{Float64},
    bu :: Vector{Float64},
    x  :: Vector{Float64},
    objfun! :: Function,
    confun! :: Function,
    optfile :: ByteString = "",
    transpose :: Bool = true,
)
    # since NAG is row-major
    #   when transpose == true the rows are linear constraints
    #   when transpose == false the columns are linear constraints
    transpose && (A = A')

    # extract various dimensions
    n = length(x)
    tda, nclin = size(A)
    ncnlin = length(bl) - n - nclin

    # check for usage problems
    length(bl) == length(bu) ||
        error("bounds vectors must have matching length")
    lexcmp(bl,bu) <= 0       ||
        error("lower bounds cannot be greater than upper bounds")
    0 <= ncnlin              ||
        error("as many bounds as variables and linear constraints must be given")
    nclin == 0 || n <= tda   ||
        error("second dimension of linear coefficients too small")

    # allocate output variables
    objf = zeros()
    g = zeros(n)

    # save callbacks in globals
    objfunref[1] = objfun!
    confunref[1] = confun!

    options = isempty(optfile) ? C_NULL :
        convert(Ptr{Void}, nag_opt_read!("e04ucc", optfile))

    reset_nag_error()
    ccall((:e04ucc, :libnagc_nag), Void,
          (NagInt, NagInt, NagInt, Ptr{Float64},
           NagInt, Ptr{Float64}, Ptr{Float64},
           Ptr{Void}, Ptr{Void},
           Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
           Ptr{Void}, Ptr{Void}, Ptr{Void}),
          n, nclin, ncnlin, A,
          tda, bl, bu,
          c_objfun, c_confun,
          x, objf, g,
          options, C_NULL, NAG_ERROR)

    return x, objf[1], g
end

const quadfunref = Array(Function)
quad_fun_wrapper(x::Float64, comm::Ptr{NagInt}) = quadfunref[1](x)::Float64
const c_quadfun = cfunction(quad_fun_wrapper, Float64, (Float64, Ptr{NagInt}))

function nag_1d_quad_inf_1(
    f :: Function,
    boundinf :: Symbol = :Infinite,
    bound :: Float64 = 0.0;
    epsabs :: Float64 = 0.0,
    epsrel :: Float64 = sqrt(eps(Float64)),
    max_num_subint :: NagInt = int32(1e7),
)
    max_num_subint > 0 || error("max num subint must be > 0")
    boundinf_val =
        boundinf == :UpperSemiInfinite ? int32(1076 + 0) :
        boundinf == :LowerSemiInfinite ? int32(1076 + 1) :
        boundinf == :Infinite          ? int32(1076 + 2) :
        error("""
        invalid boundinf symbol: $boundinf
        must be one of: UpperSemiInfinite, LowerSemiInfinite, Infinite
        """)

    # allocate output variables
    result = zeros(1)
    abserr = zeros(1)
    qp = zeros(Uint8, 48)
    comm = zeros(Uint8, 8)

    quadfunref[1] = f

    reset_nag_error()
    ccall((:d01smc, :libnagc_nag), Void,
        (Ptr{Void}, NagInt, Float64, Float64,
         Float64, NagInt, Ptr{Float64}, Ptr{Float64},
         Ptr{Void}, Ptr{Void}, Ptr{Void}),
        c_quadfun, boundinf_val, bound, epsabs,
        epsrel, max_num_subint, result, abserr,
        qp, comm, NAG_ERROR)

    return result[1], abserr[1]
end

const quadfunref_brent = Array(Function)
quad_fun_wrapper_brent(x::Float64, comm::Ptr{NagInt}) = quadfunref_brent[1](x)::Float64
const c_quadfun_brent = cfunction(quad_fun_wrapper_brent, Float64, (Float64, Ptr{NagInt}))

function nag_zero_cont_func_brent(
    a :: Float64,
    b :: Float64,
    f :: Function;
    eps :: Float64 = 1e-5,
    eta :: Float64 = 0.0,
)
    f(a)*f(b) <= 0 || error("f(a)*f(b) must be <= 0")
    quadfunref_brent[1] = f

    x = zeros(1)
    comm = zeros(Uint8, 8)

    reset_nag_error()
    ccall((:c05ayc, :libnagc_nag), Void,
        (Float64, Float64, Float64, Float64,
         Ptr{Void}, Ptr{Float64}, Ptr{Void}, Ptr{Void}),
        a, b, eps, eta, NAG.c_quadfun_brent, x, comm, NAG_ERROR)

    return x[1]
end

end # module

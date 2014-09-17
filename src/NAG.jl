module NAG
export
    nag_licence_query, nag_license_query,
    nag_complex_polygamma,
    nag_opt_read!,
    nag_opt_nlp!

nag_licence_query() = ccall((:a00acc, :libnagc_nag), Cint, ()) == 1
const nag_license_query = nag_licence_query
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

cstr_to_array(p::Ptr{Uint8}, own::Bool = false) =
    pointer_to_array(p, int(ccall(:strlen, Csize_t, (Ptr{Uint8},), p)), own)

function error_handler(msg::Ptr{Uint8}, code::Cint, name::Ptr{Uint8})
    code == 0 && return
    msg = UTF8String(copy(cstr_to_array(msg)))
    name = ASCIIString(copy(cstr_to_array(name)))
    throw(NagError(int(code),name,msg))
end

unsafe_store!(convert(Ptr{Ptr{Void}}, pointer(NAG_ERROR)) + 2*sizeof(Cint) + 512,
              cfunction(error_handler, Void, (Ptr{Uint8}, Cint, Ptr{Uint8})))

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

function nag_opt_nlp!(
    A  :: Matrix{Float64},
    bl :: Vector{Float64},
    bu :: Vector{Float64},
    x  :: Vector{Float64},
    objfun! :: Function,
    confun! :: Function,
    optfile :: ByteString = ""
)
    # extract various dimensions
    n = length(x)
    nclin, tda = size(A)
    ncnlin = length(bl) - n - nclin

    # check for usage problems
    length(bl) == length(bu) || error("bounds vectors must have matching length")
    lexcmp(bl,bu) <= 0 || error("lower bounds cannot be greater than upper bounds")
    0 <= ncnlin || error("as many bounds as variables and linear constraints must be given")
    nclin == 0 || n <= tda || error("second dimension of linear coefficients too small")

    # allocate output variables
    objf = zeros()
    g = zeros(n)

    # save callbacks in globals
    objfunref[1] = objfun!
    confunref[1] = confun!

    options = isempty(optfile) ? C_NULL :
        convert(Ptr{Void}, nag_opt_read!("e04ucc", optfile))

    fill!(NAG_ERROR, 0)
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

end # module

щч%
б%Є$
.
Abs
x"T
y"T"
Ttype:

2	
D
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
B
AssignVariableOp
resource
value"dtype"
dtypetypeИ
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
Ы
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

A
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
?
FloorMod
x"T
y"T
z"T"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
В
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
Н
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(И
?
Mul
x"T
y"T
z"T"
Ttype:
2	Р
0
Neg
x"T
y"T"
Ttype:
2
	

NoOp
U
NotEqual
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(Р
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
8
Pow
x"T
y"T
z"T"
Ttype:
2
	
e
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:
2		
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
@
Softplus
features"T
activations"T"
Ttype:
2
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	Р
┴
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring Ии
@
StaticRegexFullMatch	
input

output
"
patternstring
2
StopGradient

input"T
output"T"	
Ttype
Ў
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
М
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
Ъ
TensorListReserve
element_shape"
shape_type
num_elements#
handleКщelement_dtype"
element_dtypetype"

shape_typetype:
2	
К
TensorListSetItem
input_handle	
index
item"element_dtype*
output_handleКщelement_dtype"
element_dtypetype
И
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint         
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И
Ф
While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
И"serve*2.7.02v2.7.0-rc1-69-gc256c071bb28·Ї!
Ш
photoreceptor_rods_reike/sigmaVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*/
shared_name photoreceptor_rods_reike/sigma
С
2photoreceptor_rods_reike/sigma/Read/ReadVariableOpReadVariableOpphotoreceptor_rods_reike/sigma*
_output_shapes

:*
dtype0
к
'photoreceptor_rods_reike/sigma_scaleFacVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*8
shared_name)'photoreceptor_rods_reike/sigma_scaleFac
г
;photoreceptor_rods_reike/sigma_scaleFac/Read/ReadVariableOpReadVariableOp'photoreceptor_rods_reike/sigma_scaleFac*
_output_shapes

:*
dtype0
Ф
photoreceptor_rods_reike/phiVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*-
shared_namephotoreceptor_rods_reike/phi
Н
0photoreceptor_rods_reike/phi/Read/ReadVariableOpReadVariableOpphotoreceptor_rods_reike/phi*
_output_shapes

:*
dtype0
ж
%photoreceptor_rods_reike/phi_scaleFacVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*6
shared_name'%photoreceptor_rods_reike/phi_scaleFac
Я
9photoreceptor_rods_reike/phi_scaleFac/Read/ReadVariableOpReadVariableOp%photoreceptor_rods_reike/phi_scaleFac*
_output_shapes

:*
dtype0
Ф
photoreceptor_rods_reike/etaVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*-
shared_namephotoreceptor_rods_reike/eta
Н
0photoreceptor_rods_reike/eta/Read/ReadVariableOpReadVariableOpphotoreceptor_rods_reike/eta*
_output_shapes

:*
dtype0
ж
%photoreceptor_rods_reike/eta_scaleFacVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*6
shared_name'%photoreceptor_rods_reike/eta_scaleFac
Я
9photoreceptor_rods_reike/eta_scaleFac/Read/ReadVariableOpReadVariableOp%photoreceptor_rods_reike/eta_scaleFac*
_output_shapes

:*
dtype0
Ц
photoreceptor_rods_reike/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*.
shared_namephotoreceptor_rods_reike/beta
П
1photoreceptor_rods_reike/beta/Read/ReadVariableOpReadVariableOpphotoreceptor_rods_reike/beta*
_output_shapes

:*
dtype0
и
&photoreceptor_rods_reike/beta_scaleFacVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*7
shared_name(&photoreceptor_rods_reike/beta_scaleFac
б
:photoreceptor_rods_reike/beta_scaleFac/Read/ReadVariableOpReadVariableOp&photoreceptor_rods_reike/beta_scaleFac*
_output_shapes

:*
dtype0
Ю
!photoreceptor_rods_reike/cgmp2curVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*2
shared_name#!photoreceptor_rods_reike/cgmp2cur
Ч
5photoreceptor_rods_reike/cgmp2cur/Read/ReadVariableOpReadVariableOp!photoreceptor_rods_reike/cgmp2cur*
_output_shapes

:*
dtype0
Ю
!photoreceptor_rods_reike/cgmphillVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*2
shared_name#!photoreceptor_rods_reike/cgmphill
Ч
5photoreceptor_rods_reike/cgmphill/Read/ReadVariableOpReadVariableOp!photoreceptor_rods_reike/cgmphill*
_output_shapes

:*
dtype0
░
*photoreceptor_rods_reike/cgmphill_scaleFacVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*;
shared_name,*photoreceptor_rods_reike/cgmphill_scaleFac
й
>photoreceptor_rods_reike/cgmphill_scaleFac/Read/ReadVariableOpReadVariableOp*photoreceptor_rods_reike/cgmphill_scaleFac*
_output_shapes

:*
dtype0
Ш
photoreceptor_rods_reike/cdarkVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*/
shared_name photoreceptor_rods_reike/cdark
С
2photoreceptor_rods_reike/cdark/Read/ReadVariableOpReadVariableOpphotoreceptor_rods_reike/cdark*
_output_shapes

:*
dtype0
Ю
!photoreceptor_rods_reike/betaSlowVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*2
shared_name#!photoreceptor_rods_reike/betaSlow
Ч
5photoreceptor_rods_reike/betaSlow/Read/ReadVariableOpReadVariableOp!photoreceptor_rods_reike/betaSlow*
_output_shapes

:*
dtype0
░
*photoreceptor_rods_reike/betaSlow_scaleFacVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*;
shared_name,*photoreceptor_rods_reike/betaSlow_scaleFac
й
>photoreceptor_rods_reike/betaSlow_scaleFac/Read/ReadVariableOpReadVariableOp*photoreceptor_rods_reike/betaSlow_scaleFac*
_output_shapes

:*
dtype0
Ю
!photoreceptor_rods_reike/hillcoefVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*2
shared_name#!photoreceptor_rods_reike/hillcoef
Ч
5photoreceptor_rods_reike/hillcoef/Read/ReadVariableOpReadVariableOp!photoreceptor_rods_reike/hillcoef*
_output_shapes

:*
dtype0
░
*photoreceptor_rods_reike/hillcoef_scaleFacVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*;
shared_name,*photoreceptor_rods_reike/hillcoef_scaleFac
й
>photoreceptor_rods_reike/hillcoef_scaleFac/Read/ReadVariableOpReadVariableOp*photoreceptor_rods_reike/hillcoef_scaleFac*
_output_shapes

:*
dtype0
ж
%photoreceptor_rods_reike/hillaffinityVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*6
shared_name'%photoreceptor_rods_reike/hillaffinity
Я
9photoreceptor_rods_reike/hillaffinity/Read/ReadVariableOpReadVariableOp%photoreceptor_rods_reike/hillaffinity*
_output_shapes

:*
dtype0
╕
.photoreceptor_rods_reike/hillaffinity_scaleFacVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*?
shared_name0.photoreceptor_rods_reike/hillaffinity_scaleFac
▒
Bphotoreceptor_rods_reike/hillaffinity_scaleFac/Read/ReadVariableOpReadVariableOp.photoreceptor_rods_reike/hillaffinity_scaleFac*
_output_shapes

:*
dtype0
Ш
photoreceptor_rods_reike/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*/
shared_name photoreceptor_rods_reike/gamma
С
2photoreceptor_rods_reike/gamma/Read/ReadVariableOpReadVariableOpphotoreceptor_rods_reike/gamma*
_output_shapes

:*
dtype0
к
'photoreceptor_rods_reike/gamma_scaleFacVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*8
shared_name)'photoreceptor_rods_reike/gamma_scaleFac
г
;photoreceptor_rods_reike/gamma_scaleFac/Read/ReadVariableOpReadVariableOp'photoreceptor_rods_reike/gamma_scaleFac*
_output_shapes

:*
dtype0
Ш
photoreceptor_rods_reike/gdarkVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*/
shared_name photoreceptor_rods_reike/gdark
С
2photoreceptor_rods_reike/gdark/Read/ReadVariableOpReadVariableOpphotoreceptor_rods_reike/gdark*
_output_shapes

:*
dtype0
к
'photoreceptor_rods_reike/gdark_scaleFacVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*8
shared_name)'photoreceptor_rods_reike/gdark_scaleFac
г
;photoreceptor_rods_reike/gdark_scaleFac/Read/ReadVariableOpReadVariableOp'photoreceptor_rods_reike/gdark_scaleFac*
_output_shapes

:*
dtype0
Т
layer_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:x'**
shared_namelayer_normalization/gamma
Л
-layer_normalization/gamma/Read/ReadVariableOpReadVariableOplayer_normalization/gamma*"
_output_shapes
:x'*
dtype0
Р
layer_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:x'*)
shared_namelayer_normalization/beta
Й
,layer_normalization/beta/Read/ReadVariableOpReadVariableOplayer_normalization/beta*"
_output_shapes
:x'*
dtype0
Ж
CNNs_start/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:		x*"
shared_nameCNNs_start/kernel

%CNNs_start/kernel/Read/ReadVariableOpReadVariableOpCNNs_start/kernel*&
_output_shapes
:		x*
dtype0
v
CNNs_start/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameCNNs_start/bias
o
#CNNs_start/bias/Read/ReadVariableOpReadVariableOpCNNs_start/bias*
_output_shapes
:*
dtype0
Л
batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:╨***
shared_namebatch_normalization/gamma
Д
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes	
:╨**
dtype0
Й
batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:╨**)
shared_namebatch_normalization/beta
В
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes	
:╨**
dtype0
Ч
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:╨**0
shared_name!batch_normalization/moving_mean
Р
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes	
:╨**
dtype0
Я
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:╨**4
shared_name%#batch_normalization/moving_variance
Ш
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes	
:╨**
dtype0
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
:*
dtype0
П
batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:╨*,
shared_namebatch_normalization_1/gamma
И
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes	
:╨*
dtype0
Н
batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:╨*+
shared_namebatch_normalization_1/beta
Ж
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes	
:╨*
dtype0
Ы
!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:╨*2
shared_name#!batch_normalization_1/moving_mean
Ф
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes	
:╨*
dtype0
г
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:╨*6
shared_name'%batch_normalization_1/moving_variance
Ь
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes	
:╨*
dtype0
В
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:*
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
:*
dtype0
О
batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*,
shared_namebatch_normalization_2/gamma
З
/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes
:Z*
dtype0
М
batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*+
shared_namebatch_normalization_2/beta
Е
.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes
:Z*
dtype0
Ъ
!batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*2
shared_name#!batch_normalization_2/moving_mean
У
5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes
:Z*
dtype0
в
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*6
shared_name'%batch_normalization_2/moving_variance
Ы
9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
_output_shapes
:Z*
dtype0
О
batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*,
shared_namebatch_normalization_3/gamma
З
/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_3/gamma*
_output_shapes
:Z*
dtype0
М
batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*+
shared_namebatch_normalization_3/beta
Е
.batch_normalization_3/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_3/beta*
_output_shapes
:Z*
dtype0
Ъ
!batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*2
shared_name#!batch_normalization_3/moving_mean
У
5batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
_output_shapes
:Z*
dtype0
в
%batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*6
shared_name'%batch_normalization_3/moving_variance
Ы
9batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_3/moving_variance*
_output_shapes
:Z*
dtype0
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z%*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:Z%*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:%*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:%*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
b
total_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_3
[
total_3/Read/ReadVariableOpReadVariableOptotal_3*
_output_shapes
: *
dtype0
b
count_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_3
[
count_3/Read/ReadVariableOpReadVariableOpcount_3*
_output_shapes
: *
dtype0

NoOpNoOp
ЙК
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*├Й
value╕ЙB┤Й BмЙ
Б
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer-4
layer_with_weights-1
layer-5
layer_with_weights-2
layer-6
layer-7
	layer_with_weights-3
	layer-8

layer-9
layer-10
layer-11
layer-12
layer_with_weights-4
layer-13
layer-14
layer_with_weights-5
layer-15
layer-16
layer-17
layer-18
layer_with_weights-6
layer-19
layer-20
layer_with_weights-7
layer-21
layer-22
layer-23
layer-24
layer-25
layer_with_weights-8
layer-26
layer_with_weights-9
layer-27
layer-28
	optimizer
	variables
 trainable_variables
!regularization_losses
"	keras_api
#
signatures
 
R
$	variables
%trainable_variables
&regularization_losses
'	keras_api
╖
	(sigma
)sigma_scaleFac
*phi
+phi_scaleFac
,eta
-eta_scaleFac
.beta
/beta_scaleFac
0cgmp2cur
1cgmphill
2cgmphill_scaleFac
	3cdark
4betaSlow
5betaSlow_scaleFac
6hillcoef
7hillcoef_scaleFac
8hillaffinity
9hillaffinity_scaleFac
	:gamma
;gamma_scaleFac
	<gdark
=gdark_scaleFac
>	variables
?trainable_variables
@regularization_losses
A	keras_api
R
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api

F	keras_api
q
Gaxis
	Hgamma
Ibeta
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
h

Nkernel
Obias
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
R
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
Ч
Xaxis
	Ygamma
Zbeta
[moving_mean
\moving_variance
]	variables
^trainable_variables
_regularization_losses
`	keras_api
R
a	variables
btrainable_variables
cregularization_losses
d	keras_api
R
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
R
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
R
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
h

qkernel
rbias
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
R
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
Ы
{axis
	|gamma
}beta
~moving_mean
moving_variance
А	variables
Бtrainable_variables
Вregularization_losses
Г	keras_api
V
Д	variables
Еtrainable_variables
Жregularization_losses
З	keras_api
V
И	variables
Йtrainable_variables
Кregularization_losses
Л	keras_api
V
М	variables
Нtrainable_variables
Оregularization_losses
П	keras_api
n
Рkernel
	Сbias
Т	variables
Уtrainable_variables
Фregularization_losses
Х	keras_api
V
Ц	variables
Чtrainable_variables
Шregularization_losses
Щ	keras_api
а
	Ъaxis

Ыgamma
	Ьbeta
Эmoving_mean
Юmoving_variance
Я	variables
аtrainable_variables
бregularization_losses
в	keras_api
V
г	variables
дtrainable_variables
еregularization_losses
ж	keras_api
V
з	variables
иtrainable_variables
йregularization_losses
к	keras_api
V
л	variables
мtrainable_variables
нregularization_losses
о	keras_api
V
п	variables
░trainable_variables
▒regularization_losses
▓	keras_api
а
	│axis

┤gamma
	╡beta
╢moving_mean
╖moving_variance
╕	variables
╣trainable_variables
║regularization_losses
╗	keras_api
n
╝kernel
	╜bias
╛	variables
┐trainable_variables
└regularization_losses
┴	keras_api
V
┬	variables
├trainable_variables
─regularization_losses
┼	keras_api
 
В
(0
*1
,2
.3
)4
+5
-6
/7
08
19
210
311
412
513
614
715
816
917
:18
;19
<20
=21
H22
I23
N24
O25
Y26
Z27
[28
\29
q30
r31
|32
}33
~34
35
Р36
С37
Ы38
Ь39
Э40
Ю41
┤42
╡43
╢44
╖45
╝46
╜47
о
(0
*1
,2
.3
H4
I5
N6
O7
Y8
Z9
q10
r11
|12
}13
Р14
С15
Ы16
Ь17
┤18
╡19
╝20
╜21
 
▓
╞non_trainable_variables
╟layers
╚metrics
 ╔layer_regularization_losses
╩layer_metrics
	variables
 trainable_variables
!regularization_losses
 
 
 
 
▓
╦non_trainable_variables
╠layers
═metrics
 ╬layer_regularization_losses
╧layer_metrics
$	variables
%trainable_variables
&regularization_losses
ig
VARIABLE_VALUEphotoreceptor_rods_reike/sigma5layer_with_weights-0/sigma/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE'photoreceptor_rods_reike/sigma_scaleFac>layer_with_weights-0/sigma_scaleFac/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEphotoreceptor_rods_reike/phi3layer_with_weights-0/phi/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE%photoreceptor_rods_reike/phi_scaleFac<layer_with_weights-0/phi_scaleFac/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEphotoreceptor_rods_reike/eta3layer_with_weights-0/eta/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE%photoreceptor_rods_reike/eta_scaleFac<layer_with_weights-0/eta_scaleFac/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUEphotoreceptor_rods_reike/beta4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE&photoreceptor_rods_reike/beta_scaleFac=layer_with_weights-0/beta_scaleFac/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE!photoreceptor_rods_reike/cgmp2cur8layer_with_weights-0/cgmp2cur/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE!photoreceptor_rods_reike/cgmphill8layer_with_weights-0/cgmphill/.ATTRIBUTES/VARIABLE_VALUE
Б
VARIABLE_VALUE*photoreceptor_rods_reike/cgmphill_scaleFacAlayer_with_weights-0/cgmphill_scaleFac/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUEphotoreceptor_rods_reike/cdark5layer_with_weights-0/cdark/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE!photoreceptor_rods_reike/betaSlow8layer_with_weights-0/betaSlow/.ATTRIBUTES/VARIABLE_VALUE
Б
VARIABLE_VALUE*photoreceptor_rods_reike/betaSlow_scaleFacAlayer_with_weights-0/betaSlow_scaleFac/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE!photoreceptor_rods_reike/hillcoef8layer_with_weights-0/hillcoef/.ATTRIBUTES/VARIABLE_VALUE
Б
VARIABLE_VALUE*photoreceptor_rods_reike/hillcoef_scaleFacAlayer_with_weights-0/hillcoef_scaleFac/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE%photoreceptor_rods_reike/hillaffinity<layer_with_weights-0/hillaffinity/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE.photoreceptor_rods_reike/hillaffinity_scaleFacElayer_with_weights-0/hillaffinity_scaleFac/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUEphotoreceptor_rods_reike/gamma5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE'photoreceptor_rods_reike/gamma_scaleFac>layer_with_weights-0/gamma_scaleFac/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUEphotoreceptor_rods_reike/gdark5layer_with_weights-0/gdark/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE'photoreceptor_rods_reike/gdark_scaleFac>layer_with_weights-0/gdark_scaleFac/.ATTRIBUTES/VARIABLE_VALUE
ж
(0
*1
,2
.3
)4
+5
-6
/7
08
19
210
311
412
513
614
715
816
917
:18
;19
<20
=21

(0
*1
,2
.3
 
▓
╨non_trainable_variables
╤layers
╥metrics
 ╙layer_regularization_losses
╘layer_metrics
>	variables
?trainable_variables
@regularization_losses
 
 
 
▓
╒non_trainable_variables
╓layers
╫metrics
 ╪layer_regularization_losses
┘layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
 
 
db
VARIABLE_VALUElayer_normalization/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUElayer_normalization/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE

H0
I1

H0
I1
 
▓
┌non_trainable_variables
█layers
▄metrics
 ▌layer_regularization_losses
▐layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
][
VARIABLE_VALUECNNs_start/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUECNNs_start/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

N0
O1

N0
O1
 
▓
▀non_trainable_variables
рlayers
сmetrics
 тlayer_regularization_losses
уlayer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
 
 
 
▓
фnon_trainable_variables
хlayers
цmetrics
 чlayer_regularization_losses
шlayer_metrics
T	variables
Utrainable_variables
Vregularization_losses
 
db
VARIABLE_VALUEbatch_normalization/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbatch_normalization/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEbatch_normalization/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE#batch_normalization/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

Y0
Z1
[2
\3

Y0
Z1
 
▓
щnon_trainable_variables
ъlayers
ыmetrics
 ьlayer_regularization_losses
эlayer_metrics
]	variables
^trainable_variables
_regularization_losses
 
 
 
▓
юnon_trainable_variables
яlayers
Ёmetrics
 ёlayer_regularization_losses
Єlayer_metrics
a	variables
btrainable_variables
cregularization_losses
 
 
 
▓
єnon_trainable_variables
Їlayers
їmetrics
 Ўlayer_regularization_losses
ўlayer_metrics
e	variables
ftrainable_variables
gregularization_losses
 
 
 
▓
°non_trainable_variables
∙layers
·metrics
 √layer_regularization_losses
№layer_metrics
i	variables
jtrainable_variables
kregularization_losses
 
 
 
▓
¤non_trainable_variables
■layers
 metrics
 Аlayer_regularization_losses
Бlayer_metrics
m	variables
ntrainable_variables
oregularization_losses
YW
VARIABLE_VALUEconv2d/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

q0
r1

q0
r1
 
▓
Вnon_trainable_variables
Гlayers
Дmetrics
 Еlayer_regularization_losses
Жlayer_metrics
s	variables
ttrainable_variables
uregularization_losses
 
 
 
▓
Зnon_trainable_variables
Иlayers
Йmetrics
 Кlayer_regularization_losses
Лlayer_metrics
w	variables
xtrainable_variables
yregularization_losses
 
fd
VARIABLE_VALUEbatch_normalization_1/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_1/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_1/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_1/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

|0
}1
~2
3

|0
}1
 
╡
Мnon_trainable_variables
Нlayers
Оmetrics
 Пlayer_regularization_losses
Рlayer_metrics
А	variables
Бtrainable_variables
Вregularization_losses
 
 
 
╡
Сnon_trainable_variables
Тlayers
Уmetrics
 Фlayer_regularization_losses
Хlayer_metrics
Д	variables
Еtrainable_variables
Жregularization_losses
 
 
 
╡
Цnon_trainable_variables
Чlayers
Шmetrics
 Щlayer_regularization_losses
Ъlayer_metrics
И	variables
Йtrainable_variables
Кregularization_losses
 
 
 
╡
Ыnon_trainable_variables
Ьlayers
Эmetrics
 Юlayer_regularization_losses
Яlayer_metrics
М	variables
Нtrainable_variables
Оregularization_losses
[Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

Р0
С1

Р0
С1
 
╡
аnon_trainable_variables
бlayers
вmetrics
 гlayer_regularization_losses
дlayer_metrics
Т	variables
Уtrainable_variables
Фregularization_losses
 
 
 
╡
еnon_trainable_variables
жlayers
зmetrics
 иlayer_regularization_losses
йlayer_metrics
Ц	variables
Чtrainable_variables
Шregularization_losses
 
fd
VARIABLE_VALUEbatch_normalization_2/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_2/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_2/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_2/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
Ы0
Ь1
Э2
Ю3

Ы0
Ь1
 
╡
кnon_trainable_variables
лlayers
мmetrics
 нlayer_regularization_losses
оlayer_metrics
Я	variables
аtrainable_variables
бregularization_losses
 
 
 
╡
пnon_trainable_variables
░layers
▒metrics
 ▓layer_regularization_losses
│layer_metrics
г	variables
дtrainable_variables
еregularization_losses
 
 
 
╡
┤non_trainable_variables
╡layers
╢metrics
 ╖layer_regularization_losses
╕layer_metrics
з	variables
иtrainable_variables
йregularization_losses
 
 
 
╡
╣non_trainable_variables
║layers
╗metrics
 ╝layer_regularization_losses
╜layer_metrics
л	variables
мtrainable_variables
нregularization_losses
 
 
 
╡
╛non_trainable_variables
┐layers
└metrics
 ┴layer_regularization_losses
┬layer_metrics
п	variables
░trainable_variables
▒regularization_losses
 
fd
VARIABLE_VALUEbatch_normalization_3/gamma5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_3/beta4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_3/moving_mean;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_3/moving_variance?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
┤0
╡1
╢2
╖3

┤0
╡1
 
╡
├non_trainable_variables
─layers
┼metrics
 ╞layer_regularization_losses
╟layer_metrics
╕	variables
╣trainable_variables
║regularization_losses
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE

╝0
╜1

╝0
╜1
 
╡
╚non_trainable_variables
╔layers
╩metrics
 ╦layer_regularization_losses
╠layer_metrics
╛	variables
┐trainable_variables
└regularization_losses
 
 
 
╡
═non_trainable_variables
╬layers
╧metrics
 ╨layer_regularization_losses
╤layer_metrics
┬	variables
├trainable_variables
─regularization_losses
╩
)0
+1
-2
/3
04
15
26
37
48
59
610
711
812
913
:14
;15
<16
=17
[18
\19
~20
21
Э22
Ю23
╢24
╖25
▐
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
 
╥0
╙1
╘2
╒3
 
 
 
 
 
 
 
Ж
)0
+1
-2
/3
04
15
26
37
48
59
610
711
812
913
:14
;15
<16
=17
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

[0
\1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

~0
1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

Э0
Ю1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

╢0
╖1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

╓total

╫count
╪	variables
┘	keras_api
I

┌total

█count
▄
_fn_kwargs
▌	variables
▐	keras_api
I

▀total

рcount
с
_fn_kwargs
т	variables
у	keras_api
I

фtotal

хcount
ц
_fn_kwargs
ч	variables
ш	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

╓0
╫1

╪	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

┌0
█1

▌	variables
QO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE
 

▀0
р1

т	variables
QO
VARIABLE_VALUEtotal_34keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_34keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUE
 

ф0
х1

ч	variables
М
serving_default_input_1Placeholder*0
_output_shapes
:         ┤'*
dtype0*%
shape:         ┤'
∙
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1photoreceptor_rods_reike/sigma'photoreceptor_rods_reike/sigma_scaleFacphotoreceptor_rods_reike/phi%photoreceptor_rods_reike/phi_scaleFacphotoreceptor_rods_reike/eta%photoreceptor_rods_reike/eta_scaleFac!photoreceptor_rods_reike/cgmphill*photoreceptor_rods_reike/cgmphill_scaleFacphotoreceptor_rods_reike/beta&photoreceptor_rods_reike/beta_scaleFac!photoreceptor_rods_reike/betaSlow*photoreceptor_rods_reike/betaSlow_scaleFac!photoreceptor_rods_reike/hillcoef*photoreceptor_rods_reike/hillcoef_scaleFac%photoreceptor_rods_reike/hillaffinity.photoreceptor_rods_reike/hillaffinity_scaleFacphotoreceptor_rods_reike/gamma'photoreceptor_rods_reike/gamma_scaleFacphotoreceptor_rods_reike/gdark'photoreceptor_rods_reike/gdark_scaleFac!photoreceptor_rods_reike/cgmp2curphotoreceptor_rods_reike/cdarklayer_normalization/gammalayer_normalization/betaCNNs_start/kernelCNNs_start/bias#batch_normalization/moving_variancebatch_normalization/gammabatch_normalization/moving_meanbatch_normalization/betaconv2d/kernelconv2d/bias%batch_normalization_1/moving_variancebatch_normalization_1/gamma!batch_normalization_1/moving_meanbatch_normalization_1/betaconv2d_1/kernelconv2d_1/bias%batch_normalization_2/moving_variancebatch_normalization_2/gamma!batch_normalization_2/moving_meanbatch_normalization_2/beta%batch_normalization_3/moving_variancebatch_normalization_3/gamma!batch_normalization_3/moving_meanbatch_normalization_3/betadense/kernel
dense/bias*<
Tin5
321*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         %*R
_read_only_resource_inputs4
20	
 !"#$%&'()*+,-./0*0
config_proto 

CPU

GPU2*0J 8В *+
f&R$
"__inference_signature_wrapper_3483
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
▐
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename2photoreceptor_rods_reike/sigma/Read/ReadVariableOp;photoreceptor_rods_reike/sigma_scaleFac/Read/ReadVariableOp0photoreceptor_rods_reike/phi/Read/ReadVariableOp9photoreceptor_rods_reike/phi_scaleFac/Read/ReadVariableOp0photoreceptor_rods_reike/eta/Read/ReadVariableOp9photoreceptor_rods_reike/eta_scaleFac/Read/ReadVariableOp1photoreceptor_rods_reike/beta/Read/ReadVariableOp:photoreceptor_rods_reike/beta_scaleFac/Read/ReadVariableOp5photoreceptor_rods_reike/cgmp2cur/Read/ReadVariableOp5photoreceptor_rods_reike/cgmphill/Read/ReadVariableOp>photoreceptor_rods_reike/cgmphill_scaleFac/Read/ReadVariableOp2photoreceptor_rods_reike/cdark/Read/ReadVariableOp5photoreceptor_rods_reike/betaSlow/Read/ReadVariableOp>photoreceptor_rods_reike/betaSlow_scaleFac/Read/ReadVariableOp5photoreceptor_rods_reike/hillcoef/Read/ReadVariableOp>photoreceptor_rods_reike/hillcoef_scaleFac/Read/ReadVariableOp9photoreceptor_rods_reike/hillaffinity/Read/ReadVariableOpBphotoreceptor_rods_reike/hillaffinity_scaleFac/Read/ReadVariableOp2photoreceptor_rods_reike/gamma/Read/ReadVariableOp;photoreceptor_rods_reike/gamma_scaleFac/Read/ReadVariableOp2photoreceptor_rods_reike/gdark/Read/ReadVariableOp;photoreceptor_rods_reike/gdark_scaleFac/Read/ReadVariableOp-layer_normalization/gamma/Read/ReadVariableOp,layer_normalization/beta/Read/ReadVariableOp%CNNs_start/kernel/Read/ReadVariableOp#CNNs_start/bias/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp/batch_normalization_2/gamma/Read/ReadVariableOp.batch_normalization_2/beta/Read/ReadVariableOp5batch_normalization_2/moving_mean/Read/ReadVariableOp9batch_normalization_2/moving_variance/Read/ReadVariableOp/batch_normalization_3/gamma/Read/ReadVariableOp.batch_normalization_3/beta/Read/ReadVariableOp5batch_normalization_3/moving_mean/Read/ReadVariableOp9batch_normalization_3/moving_variance/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOptotal_3/Read/ReadVariableOpcount_3/Read/ReadVariableOpConst*E
Tin>
<2:*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *&
f!R
__inference__traced_save_5420
∙
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamephotoreceptor_rods_reike/sigma'photoreceptor_rods_reike/sigma_scaleFacphotoreceptor_rods_reike/phi%photoreceptor_rods_reike/phi_scaleFacphotoreceptor_rods_reike/eta%photoreceptor_rods_reike/eta_scaleFacphotoreceptor_rods_reike/beta&photoreceptor_rods_reike/beta_scaleFac!photoreceptor_rods_reike/cgmp2cur!photoreceptor_rods_reike/cgmphill*photoreceptor_rods_reike/cgmphill_scaleFacphotoreceptor_rods_reike/cdark!photoreceptor_rods_reike/betaSlow*photoreceptor_rods_reike/betaSlow_scaleFac!photoreceptor_rods_reike/hillcoef*photoreceptor_rods_reike/hillcoef_scaleFac%photoreceptor_rods_reike/hillaffinity.photoreceptor_rods_reike/hillaffinity_scaleFacphotoreceptor_rods_reike/gamma'photoreceptor_rods_reike/gamma_scaleFacphotoreceptor_rods_reike/gdark'photoreceptor_rods_reike/gdark_scaleFaclayer_normalization/gammalayer_normalization/betaCNNs_start/kernelCNNs_start/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv2d/kernelconv2d/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv2d_1/kernelconv2d_1/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_variancebatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_variancedense/kernel
dense/biastotalcounttotal_1count_1total_2count_2total_3count_3*D
Tin=
;29*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *)
f$R"
 __inference__traced_restore_5598Щ╖
жЇ
Ъ-
I__inference_PRFR_CNN2D_RODS_layer_call_and_return_conditional_losses_3961

inputsB
0photoreceptor_rods_reike_readvariableop_resource:F
4photoreceptor_rods_reike_mul_readvariableop_resource:D
2photoreceptor_rods_reike_readvariableop_1_resource:H
6photoreceptor_rods_reike_mul_1_readvariableop_resource:D
2photoreceptor_rods_reike_readvariableop_2_resource:H
6photoreceptor_rods_reike_mul_2_readvariableop_resource:D
2photoreceptor_rods_reike_readvariableop_3_resource:H
6photoreceptor_rods_reike_mul_3_readvariableop_resource:D
2photoreceptor_rods_reike_readvariableop_4_resource:H
6photoreceptor_rods_reike_mul_4_readvariableop_resource:D
2photoreceptor_rods_reike_readvariableop_5_resource:H
6photoreceptor_rods_reike_mul_5_readvariableop_resource:D
2photoreceptor_rods_reike_readvariableop_6_resource:H
6photoreceptor_rods_reike_mul_6_readvariableop_resource:D
2photoreceptor_rods_reike_readvariableop_7_resource:H
6photoreceptor_rods_reike_mul_7_readvariableop_resource:D
2photoreceptor_rods_reike_readvariableop_8_resource:H
6photoreceptor_rods_reike_mul_8_readvariableop_resource:D
2photoreceptor_rods_reike_readvariableop_9_resource:H
6photoreceptor_rods_reike_mul_9_readvariableop_resource:/
photoreceptor_rods_reike_3751:/
photoreceptor_rods_reike_3753:I
3layer_normalization_reshape_readvariableop_resource:x'K
5layer_normalization_reshape_1_readvariableop_resource:x'C
)cnns_start_conv2d_readvariableop_resource:		x8
*cnns_start_biasadd_readvariableop_resource:D
5batch_normalization_batchnorm_readvariableop_resource:	╨*H
9batch_normalization_batchnorm_mul_readvariableop_resource:	╨*F
7batch_normalization_batchnorm_readvariableop_1_resource:	╨*F
7batch_normalization_batchnorm_readvariableop_2_resource:	╨*?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:F
7batch_normalization_1_batchnorm_readvariableop_resource:	╨J
;batch_normalization_1_batchnorm_mul_readvariableop_resource:	╨H
9batch_normalization_1_batchnorm_readvariableop_1_resource:	╨H
9batch_normalization_1_batchnorm_readvariableop_2_resource:	╨A
'conv2d_1_conv2d_readvariableop_resource:6
(conv2d_1_biasadd_readvariableop_resource:E
7batch_normalization_2_batchnorm_readvariableop_resource:ZI
;batch_normalization_2_batchnorm_mul_readvariableop_resource:ZG
9batch_normalization_2_batchnorm_readvariableop_1_resource:ZG
9batch_normalization_2_batchnorm_readvariableop_2_resource:ZE
7batch_normalization_3_batchnorm_readvariableop_resource:ZI
;batch_normalization_3_batchnorm_mul_readvariableop_resource:ZG
9batch_normalization_3_batchnorm_readvariableop_1_resource:ZG
9batch_normalization_3_batchnorm_readvariableop_2_resource:Z6
$dense_matmul_readvariableop_resource:Z%3
%dense_biasadd_readvariableop_resource:%
identity

identity_1Ив!CNNs_start/BiasAdd/ReadVariableOpв CNNs_start/Conv2D/ReadVariableOpв3CNNs_start/kernel/Regularizer/Square/ReadVariableOpв,batch_normalization/batchnorm/ReadVariableOpв.batch_normalization/batchnorm/ReadVariableOp_1в.batch_normalization/batchnorm/ReadVariableOp_2в0batch_normalization/batchnorm/mul/ReadVariableOpв.batch_normalization_1/batchnorm/ReadVariableOpв0batch_normalization_1/batchnorm/ReadVariableOp_1в0batch_normalization_1/batchnorm/ReadVariableOp_2в2batch_normalization_1/batchnorm/mul/ReadVariableOpв.batch_normalization_2/batchnorm/ReadVariableOpв0batch_normalization_2/batchnorm/ReadVariableOp_1в0batch_normalization_2/batchnorm/ReadVariableOp_2в2batch_normalization_2/batchnorm/mul/ReadVariableOpв.batch_normalization_3/batchnorm/ReadVariableOpв0batch_normalization_3/batchnorm/ReadVariableOp_1в0batch_normalization_3/batchnorm/ReadVariableOp_2в2batch_normalization_3/batchnorm/mul/ReadVariableOpвconv2d/BiasAdd/ReadVariableOpвconv2d/Conv2D/ReadVariableOpв/conv2d/kernel/Regularizer/Square/ReadVariableOpвconv2d_1/BiasAdd/ReadVariableOpвconv2d_1/Conv2D/ReadVariableOpв1conv2d_1/kernel/Regularizer/Square/ReadVariableOpвdense/BiasAdd/ReadVariableOpвdense/MatMul/ReadVariableOpв.dense/kernel/Regularizer/Square/ReadVariableOpв*layer_normalization/Reshape/ReadVariableOpв,layer_normalization/Reshape_1/ReadVariableOpв'photoreceptor_rods_reike/ReadVariableOpв)photoreceptor_rods_reike/ReadVariableOp_1в)photoreceptor_rods_reike/ReadVariableOp_2в)photoreceptor_rods_reike/ReadVariableOp_3в)photoreceptor_rods_reike/ReadVariableOp_4в)photoreceptor_rods_reike/ReadVariableOp_5в)photoreceptor_rods_reike/ReadVariableOp_6в)photoreceptor_rods_reike/ReadVariableOp_7в)photoreceptor_rods_reike/ReadVariableOp_8в)photoreceptor_rods_reike/ReadVariableOp_9в0photoreceptor_rods_reike/StatefulPartitionedCallв+photoreceptor_rods_reike/mul/ReadVariableOpв-photoreceptor_rods_reike/mul_1/ReadVariableOpв-photoreceptor_rods_reike/mul_2/ReadVariableOpв-photoreceptor_rods_reike/mul_3/ReadVariableOpв-photoreceptor_rods_reike/mul_4/ReadVariableOpв-photoreceptor_rods_reike/mul_5/ReadVariableOpв-photoreceptor_rods_reike/mul_6/ReadVariableOpв-photoreceptor_rods_reike/mul_7/ReadVariableOpв-photoreceptor_rods_reike/mul_8/ReadVariableOpв-photoreceptor_rods_reike/mul_9/ReadVariableOpC
reshape/ShapeShapeinputs*
T0*
_output_shapes
:e
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:∙
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :┤Z
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :Т	п
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:z
reshape/ReshapeReshapeinputsreshape/Reshape/shape:output:0*
T0*-
_output_shapes
:         ┤Т	Ш
'photoreceptor_rods_reike/ReadVariableOpReadVariableOp0photoreceptor_rods_reike_readvariableop_resource*
_output_shapes

:*
dtype0а
+photoreceptor_rods_reike/mul/ReadVariableOpReadVariableOp4photoreceptor_rods_reike_mul_readvariableop_resource*
_output_shapes

:*
dtype0▓
photoreceptor_rods_reike/mulMul/photoreceptor_rods_reike/ReadVariableOp:value:03photoreceptor_rods_reike/mul/ReadVariableOp:value:0*
T0*
_output_shapes

:Ь
)photoreceptor_rods_reike/ReadVariableOp_1ReadVariableOp2photoreceptor_rods_reike_readvariableop_1_resource*
_output_shapes

:*
dtype0д
-photoreceptor_rods_reike/mul_1/ReadVariableOpReadVariableOp6photoreceptor_rods_reike_mul_1_readvariableop_resource*
_output_shapes

:*
dtype0╕
photoreceptor_rods_reike/mul_1Mul1photoreceptor_rods_reike/ReadVariableOp_1:value:05photoreceptor_rods_reike/mul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:Ь
)photoreceptor_rods_reike/ReadVariableOp_2ReadVariableOp2photoreceptor_rods_reike_readvariableop_2_resource*
_output_shapes

:*
dtype0д
-photoreceptor_rods_reike/mul_2/ReadVariableOpReadVariableOp6photoreceptor_rods_reike_mul_2_readvariableop_resource*
_output_shapes

:*
dtype0╕
photoreceptor_rods_reike/mul_2Mul1photoreceptor_rods_reike/ReadVariableOp_2:value:05photoreceptor_rods_reike/mul_2/ReadVariableOp:value:0*
T0*
_output_shapes

:Ь
)photoreceptor_rods_reike/ReadVariableOp_3ReadVariableOp2photoreceptor_rods_reike_readvariableop_3_resource*
_output_shapes

:*
dtype0д
-photoreceptor_rods_reike/mul_3/ReadVariableOpReadVariableOp6photoreceptor_rods_reike_mul_3_readvariableop_resource*
_output_shapes

:*
dtype0╕
photoreceptor_rods_reike/mul_3Mul1photoreceptor_rods_reike/ReadVariableOp_3:value:05photoreceptor_rods_reike/mul_3/ReadVariableOp:value:0*
T0*
_output_shapes

:Ь
)photoreceptor_rods_reike/ReadVariableOp_4ReadVariableOp2photoreceptor_rods_reike_readvariableop_4_resource*
_output_shapes

:*
dtype0д
-photoreceptor_rods_reike/mul_4/ReadVariableOpReadVariableOp6photoreceptor_rods_reike_mul_4_readvariableop_resource*
_output_shapes

:*
dtype0╕
photoreceptor_rods_reike/mul_4Mul1photoreceptor_rods_reike/ReadVariableOp_4:value:05photoreceptor_rods_reike/mul_4/ReadVariableOp:value:0*
T0*
_output_shapes

:Ь
)photoreceptor_rods_reike/ReadVariableOp_5ReadVariableOp2photoreceptor_rods_reike_readvariableop_5_resource*
_output_shapes

:*
dtype0д
-photoreceptor_rods_reike/mul_5/ReadVariableOpReadVariableOp6photoreceptor_rods_reike_mul_5_readvariableop_resource*
_output_shapes

:*
dtype0╕
photoreceptor_rods_reike/mul_5Mul1photoreceptor_rods_reike/ReadVariableOp_5:value:05photoreceptor_rods_reike/mul_5/ReadVariableOp:value:0*
T0*
_output_shapes

:Ь
)photoreceptor_rods_reike/ReadVariableOp_6ReadVariableOp2photoreceptor_rods_reike_readvariableop_6_resource*
_output_shapes

:*
dtype0д
-photoreceptor_rods_reike/mul_6/ReadVariableOpReadVariableOp6photoreceptor_rods_reike_mul_6_readvariableop_resource*
_output_shapes

:*
dtype0╕
photoreceptor_rods_reike/mul_6Mul1photoreceptor_rods_reike/ReadVariableOp_6:value:05photoreceptor_rods_reike/mul_6/ReadVariableOp:value:0*
T0*
_output_shapes

:Ь
)photoreceptor_rods_reike/ReadVariableOp_7ReadVariableOp2photoreceptor_rods_reike_readvariableop_7_resource*
_output_shapes

:*
dtype0д
-photoreceptor_rods_reike/mul_7/ReadVariableOpReadVariableOp6photoreceptor_rods_reike_mul_7_readvariableop_resource*
_output_shapes

:*
dtype0╕
photoreceptor_rods_reike/mul_7Mul1photoreceptor_rods_reike/ReadVariableOp_7:value:05photoreceptor_rods_reike/mul_7/ReadVariableOp:value:0*
T0*
_output_shapes

:Ь
)photoreceptor_rods_reike/ReadVariableOp_8ReadVariableOp2photoreceptor_rods_reike_readvariableop_8_resource*
_output_shapes

:*
dtype0д
-photoreceptor_rods_reike/mul_8/ReadVariableOpReadVariableOp6photoreceptor_rods_reike_mul_8_readvariableop_resource*
_output_shapes

:*
dtype0╕
photoreceptor_rods_reike/mul_8Mul1photoreceptor_rods_reike/ReadVariableOp_8:value:05photoreceptor_rods_reike/mul_8/ReadVariableOp:value:0*
T0*
_output_shapes

:g
"photoreceptor_rods_reike/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Aе
 photoreceptor_rods_reike/truedivRealDiv"photoreceptor_rods_reike/mul_8:z:0+photoreceptor_rods_reike/truediv/y:output:0*
T0*
_output_shapes

:Ь
)photoreceptor_rods_reike/ReadVariableOp_9ReadVariableOp2photoreceptor_rods_reike_readvariableop_9_resource*
_output_shapes

:*
dtype0д
-photoreceptor_rods_reike/mul_9/ReadVariableOpReadVariableOp6photoreceptor_rods_reike_mul_9_readvariableop_resource*
_output_shapes

:*
dtype0╕
photoreceptor_rods_reike/mul_9Mul1photoreceptor_rods_reike/ReadVariableOp_9:value:05photoreceptor_rods_reike/mul_9/ReadVariableOp:value:0*
T0*
_output_shapes

:А
0photoreceptor_rods_reike/StatefulPartitionedCallStatefulPartitionedCallreshape/Reshape:output:0 photoreceptor_rods_reike/mul:z:0"photoreceptor_rods_reike/mul_1:z:0"photoreceptor_rods_reike/mul_2:z:0photoreceptor_rods_reike_3751"photoreceptor_rods_reike/mul_3:z:0photoreceptor_rods_reike_3753"photoreceptor_rods_reike/mul_4:z:0"photoreceptor_rods_reike/mul_5:z:0"photoreceptor_rods_reike/mul_6:z:0"photoreceptor_rods_reike/mul_7:z:0$photoreceptor_rods_reike/truediv:z:0"photoreceptor_rods_reike/mul_9:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         ┤Т	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *$
fR
__inference_riekeModel_1187x
reshape_1/ShapeShape9photoreceptor_rods_reike/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:g
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Г
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :┤[
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :'█
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0"reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:┤
reshape_1/ReshapeReshape9photoreceptor_rods_reike/StatefulPartitionedCall:output:0 reshape_1/Reshape/shape:output:0*
T0*0
_output_shapes
:         ┤'Е
,tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"    <           З
.tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                З
.tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            ф
&tf.__operators__.getitem/strided_sliceStridedSlicereshape_1/Reshape:output:05tf.__operators__.getitem/strided_slice/stack:output:07tf.__operators__.getitem/strided_slice/stack_1:output:07tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         x'*

begin_mask*
end_maskЗ
2layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         с
 layer_normalization/moments/meanMean/tf.__operators__.getitem/strided_slice:output:0;layer_normalization/moments/mean/reduction_indices:output:0*
T0*/
_output_shapes
:         *
	keep_dims(Э
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*/
_output_shapes
:         р
-layer_normalization/moments/SquaredDifferenceSquaredDifference/tf.__operators__.getitem/strided_slice:output:01layer_normalization/moments/StopGradient:output:0*
T0*/
_output_shapes
:         x'Л
6layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         ы
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
T0*/
_output_shapes
:         *
	keep_dims(в
*layer_normalization/Reshape/ReadVariableOpReadVariableOp3layer_normalization_reshape_readvariableop_resource*"
_output_shapes
:x'*
dtype0z
!layer_normalization/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   x      '   ╖
layer_normalization/ReshapeReshape2layer_normalization/Reshape/ReadVariableOp:value:0*layer_normalization/Reshape/shape:output:0*
T0*&
_output_shapes
:x'ж
,layer_normalization/Reshape_1/ReadVariableOpReadVariableOp5layer_normalization_reshape_1_readvariableop_resource*"
_output_shapes
:x'*
dtype0|
#layer_normalization/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   x      '   ╜
layer_normalization/Reshape_1Reshape4layer_normalization/Reshape_1/ReadVariableOp:value:0,layer_normalization/Reshape_1/shape:output:0*
T0*&
_output_shapes
:x'h
#layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Х┐╓3┴
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*/
_output_shapes
:         Н
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*/
_output_shapes
:         ▒
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:0$layer_normalization/Reshape:output:0*
T0*/
_output_shapes
:         x'╝
#layer_normalization/batchnorm/mul_1Mul/tf.__operators__.getitem/strided_slice:output:0%layer_normalization/batchnorm/mul:z:0*
T0*/
_output_shapes
:         x'╢
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*/
_output_shapes
:         x'│
!layer_normalization/batchnorm/subSub&layer_normalization/Reshape_1:output:0'layer_normalization/batchnorm/mul_2:z:0*
T0*/
_output_shapes
:         x'╢
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*/
_output_shapes
:         x'Т
 CNNs_start/Conv2D/ReadVariableOpReadVariableOp)cnns_start_conv2d_readvariableop_resource*&
_output_shapes
:		x*
dtype0ш
CNNs_start/Conv2DConv2D'layer_normalization/batchnorm/add_1:z:0(CNNs_start/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
data_formatNCHW*
paddingVALID*
strides
И
!CNNs_start/BiasAdd/ReadVariableOpReadVariableOp*cnns_start_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0╡
CNNs_start/BiasAddBiasAddCNNs_start/Conv2D:output:0)CNNs_start/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
data_formatNCHW^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"    P  В
flatten/ReshapeReshapeCNNs_start/BiasAdd:output:0flatten/Const:output:0*
T0*(
_output_shapes
:         ╨*Я
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:╨**
dtype0h
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:┤
!batch_normalization/batchnorm/addAddV24batch_normalization/batchnorm/ReadVariableOp:value:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:╨*y
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:╨*з
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:╨**
dtype0▒
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:╨*Ю
#batch_normalization/batchnorm/mul_1Mulflatten/Reshape:output:0%batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:         ╨*г
.batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOp7batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes	
:╨**
dtype0п
#batch_normalization/batchnorm/mul_2Mul6batch_normalization/batchnorm/ReadVariableOp_1:value:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:╨*г
.batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOp7batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes	
:╨**
dtype0п
!batch_normalization/batchnorm/subSub6batch_normalization/batchnorm/ReadVariableOp_2:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:╨*п
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:         ╨*f
reshape_2/ShapeShape'batch_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:g
reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Г
reshape_2/strided_sliceStridedSlicereshape_2/Shape:output:0&reshape_2/strided_slice/stack:output:0(reshape_2/strided_slice/stack_1:output:0(reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :█
reshape_2/Reshape/shapePack reshape_2/strided_slice:output:0"reshape_2/Reshape/shape/1:output:0"reshape_2/Reshape/shape/2:output:0"reshape_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:б
reshape_2/ReshapeReshape'batch_normalization/batchnorm/add_1:z:0 reshape_2/Reshape/shape:output:0*
T0*/
_output_shapes
:         └
max_pooling2d/MaxPoolMaxPoolreshape_2/Reshape:output:0*/
_output_shapes
:         *
data_formatNCHW*
ksize
*
paddingVALID*
strides
q
activation/ReluRelumax_pooling2d/MaxPool:output:0*
T0*/
_output_shapes
:         К
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0╓
conv2d/Conv2DConv2Dactivation/Relu:activations:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         	*
data_formatNCHW*
paddingVALID*
strides
А
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0й
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         	*
data_formatNCHW`
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"    ╨  В
flatten_1/ReshapeReshapeconv2d/BiasAdd:output:0flatten_1/Const:output:0*
T0*(
_output_shapes
:         ╨г
.batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:╨*
dtype0j
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:║
#batch_normalization_1/batchnorm/addAddV26batch_normalization_1/batchnorm/ReadVariableOp:value:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:╨}
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:╨л
2batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:╨*
dtype0╖
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:0:batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:╨д
%batch_normalization_1/batchnorm/mul_1Mulflatten_1/Reshape:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:         ╨з
0batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes	
:╨*
dtype0╡
%batch_normalization_1/batchnorm/mul_2Mul8batch_normalization_1/batchnorm/ReadVariableOp_1:value:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:╨з
0batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes	
:╨*
dtype0╡
#batch_normalization_1/batchnorm/subSub8batch_normalization_1/batchnorm/ReadVariableOp_2:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:╨╡
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:         ╨h
reshape_3/ShapeShape)batch_normalization_1/batchnorm/add_1:z:0*
T0*
_output_shapes
:g
reshape_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Г
reshape_3/strided_sliceStridedSlicereshape_3/Shape:output:0&reshape_3/strided_slice/stack:output:0(reshape_3/strided_slice/stack_1:output:0(reshape_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
reshape_3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_3/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_3/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :	█
reshape_3/Reshape/shapePack reshape_3/strided_slice:output:0"reshape_3/Reshape/shape/1:output:0"reshape_3/Reshape/shape/2:output:0"reshape_3/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:г
reshape_3/ReshapeReshape)batch_normalization_1/batchnorm/add_1:z:0 reshape_3/Reshape/shape:output:0*
T0*/
_output_shapes
:         	o
activation_1/ReluRelureshape_3/Reshape:output:0*
T0*/
_output_shapes
:         	О
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0▄
conv2d_1/Conv2DConv2Dactivation_1/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
data_formatNCHW*
paddingVALID*
strides
Д
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0п
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
data_formatNCHW`
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"    Z   Г
flatten_2/ReshapeReshapeconv2d_1/BiasAdd:output:0flatten_2/Const:output:0*
T0*'
_output_shapes
:         Zв
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:Z*
dtype0j
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:╣
#batch_normalization_2/batchnorm/addAddV26batch_normalization_2/batchnorm/ReadVariableOp:value:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:Z|
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:Zк
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:Z*
dtype0╢
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Zг
%batch_normalization_2/batchnorm/mul_1Mulflatten_2/Reshape:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:         Zж
0batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes
:Z*
dtype0┤
%batch_normalization_2/batchnorm/mul_2Mul8batch_normalization_2/batchnorm/ReadVariableOp_1:value:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:Zж
0batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes
:Z*
dtype0┤
#batch_normalization_2/batchnorm/subSub8batch_normalization_2/batchnorm/ReadVariableOp_2:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Z┤
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:         Zh
reshape_4/ShapeShape)batch_normalization_2/batchnorm/add_1:z:0*
T0*
_output_shapes
:g
reshape_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Г
reshape_4/strided_sliceStridedSlicereshape_4/Shape:output:0&reshape_4/strided_slice/stack:output:0(reshape_4/strided_slice/stack_1:output:0(reshape_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
reshape_4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_4/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_4/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :█
reshape_4/Reshape/shapePack reshape_4/strided_slice:output:0"reshape_4/Reshape/shape/1:output:0"reshape_4/Reshape/shape/2:output:0"reshape_4/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:г
reshape_4/ReshapeReshape)batch_normalization_2/batchnorm/add_1:z:0 reshape_4/Reshape/shape:output:0*
T0*/
_output_shapes
:         o
activation_2/ReluRelureshape_4/Reshape:output:0*
T0*/
_output_shapes
:         `
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"    Z   Й
flatten_3/ReshapeReshapeactivation_2/Relu:activations:0flatten_3/Const:output:0*
T0*'
_output_shapes
:         Zв
.batch_normalization_3/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:Z*
dtype0j
%batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:╣
#batch_normalization_3/batchnorm/addAddV26batch_normalization_3/batchnorm/ReadVariableOp:value:0.batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
:Z|
%batch_normalization_3/batchnorm/RsqrtRsqrt'batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:Zк
2batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:Z*
dtype0╢
#batch_normalization_3/batchnorm/mulMul)batch_normalization_3/batchnorm/Rsqrt:y:0:batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Zг
%batch_normalization_3/batchnorm/mul_1Mulflatten_3/Reshape:output:0'batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:         Zж
0batch_normalization_3/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_3_batchnorm_readvariableop_1_resource*
_output_shapes
:Z*
dtype0┤
%batch_normalization_3/batchnorm/mul_2Mul8batch_normalization_3/batchnorm/ReadVariableOp_1:value:0'batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:Zж
0batch_normalization_3/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_3_batchnorm_readvariableop_2_resource*
_output_shapes
:Z*
dtype0┤
#batch_normalization_3/batchnorm/subSub8batch_normalization_3/batchnorm/ReadVariableOp_2:value:0)batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Z┤
%batch_normalization_3/batchnorm/add_1AddV2)batch_normalization_3/batchnorm/mul_1:z:0'batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:         ZА
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:Z%*
dtype0Ш
dense/MatMulMatMul)batch_normalization_3/batchnorm/add_1:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         %~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:%*
dtype0И
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         %n
dense/ActivityRegularizer/AbsAbsdense/BiasAdd:output:0*
T0*'
_output_shapes
:         %p
dense/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Т
dense/ActivityRegularizer/SumSum!dense/ActivityRegularizer/Abs:y:0(dense/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: d
dense/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:Ч
dense/ActivityRegularizer/mulMul(dense/ActivityRegularizer/mul/x:output:0&dense/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: e
dense/ActivityRegularizer/ShapeShapedense/BiasAdd:output:0*
T0*
_output_shapes
:w
-dense/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/dense/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/dense/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╙
'dense/ActivityRegularizer/strided_sliceStridedSlice(dense/ActivityRegularizer/Shape:output:06dense/ActivityRegularizer/strided_slice/stack:output:08dense/ActivityRegularizer/strided_slice/stack_1:output:08dense/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskИ
dense/ActivityRegularizer/CastCast0dense/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: Ф
!dense/ActivityRegularizer/truedivRealDiv!dense/ActivityRegularizer/mul:z:0"dense/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: k
activation_3/SoftplusSoftplusdense/BiasAdd:output:0*
T0*'
_output_shapes
:         %е
3CNNs_start/kernel/Regularizer/Square/ReadVariableOpReadVariableOp)cnns_start_conv2d_readvariableop_resource*&
_output_shapes
:		x*
dtype0Ь
$CNNs_start/kernel/Regularizer/SquareSquare;CNNs_start/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		x|
#CNNs_start/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             б
!CNNs_start/kernel/Regularizer/SumSum(CNNs_start/kernel/Regularizer/Square:y:0,CNNs_start/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#CNNs_start/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:г
!CNNs_start/kernel/Regularizer/mulMul,CNNs_start/kernel/Regularizer/mul/x:output:0*CNNs_start/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: Э
/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ф
 conv2d/kernel/Regularizer/SquareSquare7conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:x
conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             Х
conv2d/kernel/Regularizer/SumSum$conv2d/kernel/Regularizer/Square:y:0(conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:Ч
conv2d/kernel/Regularizer/mulMul(conv2d/kernel/Regularizer/mul/x:output:0&conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: б
1conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ш
"conv2d_1/kernel/Regularizer/SquareSquare9conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
!conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             Ы
conv2d_1/kernel/Regularizer/SumSum&conv2d_1/kernel/Regularizer/Square:y:0*conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:Э
conv2d_1/kernel/Regularizer/mulMul*conv2d_1/kernel/Regularizer/mul/x:output:0(conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: У
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:Z%*
dtype0К
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:Z%o
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Т
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:Ф
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: r
IdentityIdentity#activation_3/Softplus:activations:0^NoOp*
T0*'
_output_shapes
:         %e

Identity_1Identity%dense/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: ф
NoOpNoOp"^CNNs_start/BiasAdd/ReadVariableOp!^CNNs_start/Conv2D/ReadVariableOp4^CNNs_start/kernel/Regularizer/Square/ReadVariableOp-^batch_normalization/batchnorm/ReadVariableOp/^batch_normalization/batchnorm/ReadVariableOp_1/^batch_normalization/batchnorm/ReadVariableOp_21^batch_normalization/batchnorm/mul/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp1^batch_normalization_1/batchnorm/ReadVariableOp_11^batch_normalization_1/batchnorm/ReadVariableOp_23^batch_normalization_1/batchnorm/mul/ReadVariableOp/^batch_normalization_2/batchnorm/ReadVariableOp1^batch_normalization_2/batchnorm/ReadVariableOp_11^batch_normalization_2/batchnorm/ReadVariableOp_23^batch_normalization_2/batchnorm/mul/ReadVariableOp/^batch_normalization_3/batchnorm/ReadVariableOp1^batch_normalization_3/batchnorm/ReadVariableOp_11^batch_normalization_3/batchnorm/ReadVariableOp_23^batch_normalization_3/batchnorm/mul/ReadVariableOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp0^conv2d/kernel/Regularizer/Square/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp2^conv2d_1/kernel/Regularizer/Square/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp+^layer_normalization/Reshape/ReadVariableOp-^layer_normalization/Reshape_1/ReadVariableOp(^photoreceptor_rods_reike/ReadVariableOp*^photoreceptor_rods_reike/ReadVariableOp_1*^photoreceptor_rods_reike/ReadVariableOp_2*^photoreceptor_rods_reike/ReadVariableOp_3*^photoreceptor_rods_reike/ReadVariableOp_4*^photoreceptor_rods_reike/ReadVariableOp_5*^photoreceptor_rods_reike/ReadVariableOp_6*^photoreceptor_rods_reike/ReadVariableOp_7*^photoreceptor_rods_reike/ReadVariableOp_8*^photoreceptor_rods_reike/ReadVariableOp_91^photoreceptor_rods_reike/StatefulPartitionedCall,^photoreceptor_rods_reike/mul/ReadVariableOp.^photoreceptor_rods_reike/mul_1/ReadVariableOp.^photoreceptor_rods_reike/mul_2/ReadVariableOp.^photoreceptor_rods_reike/mul_3/ReadVariableOp.^photoreceptor_rods_reike/mul_4/ReadVariableOp.^photoreceptor_rods_reike/mul_5/ReadVariableOp.^photoreceptor_rods_reike/mul_6/ReadVariableOp.^photoreceptor_rods_reike/mul_7/ReadVariableOp.^photoreceptor_rods_reike/mul_8/ReadVariableOp.^photoreceptor_rods_reike/mul_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*П
_input_shapes~
|:         ┤': : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2F
!CNNs_start/BiasAdd/ReadVariableOp!CNNs_start/BiasAdd/ReadVariableOp2D
 CNNs_start/Conv2D/ReadVariableOp CNNs_start/Conv2D/ReadVariableOp2j
3CNNs_start/kernel/Regularizer/Square/ReadVariableOp3CNNs_start/kernel/Regularizer/Square/ReadVariableOp2\
,batch_normalization/batchnorm/ReadVariableOp,batch_normalization/batchnorm/ReadVariableOp2`
.batch_normalization/batchnorm/ReadVariableOp_1.batch_normalization/batchnorm/ReadVariableOp_12`
.batch_normalization/batchnorm/ReadVariableOp_2.batch_normalization/batchnorm/ReadVariableOp_22d
0batch_normalization/batchnorm/mul/ReadVariableOp0batch_normalization/batchnorm/mul/ReadVariableOp2`
.batch_normalization_1/batchnorm/ReadVariableOp.batch_normalization_1/batchnorm/ReadVariableOp2d
0batch_normalization_1/batchnorm/ReadVariableOp_10batch_normalization_1/batchnorm/ReadVariableOp_12d
0batch_normalization_1/batchnorm/ReadVariableOp_20batch_normalization_1/batchnorm/ReadVariableOp_22h
2batch_normalization_1/batchnorm/mul/ReadVariableOp2batch_normalization_1/batchnorm/mul/ReadVariableOp2`
.batch_normalization_2/batchnorm/ReadVariableOp.batch_normalization_2/batchnorm/ReadVariableOp2d
0batch_normalization_2/batchnorm/ReadVariableOp_10batch_normalization_2/batchnorm/ReadVariableOp_12d
0batch_normalization_2/batchnorm/ReadVariableOp_20batch_normalization_2/batchnorm/ReadVariableOp_22h
2batch_normalization_2/batchnorm/mul/ReadVariableOp2batch_normalization_2/batchnorm/mul/ReadVariableOp2`
.batch_normalization_3/batchnorm/ReadVariableOp.batch_normalization_3/batchnorm/ReadVariableOp2d
0batch_normalization_3/batchnorm/ReadVariableOp_10batch_normalization_3/batchnorm/ReadVariableOp_12d
0batch_normalization_3/batchnorm/ReadVariableOp_20batch_normalization_3/batchnorm/ReadVariableOp_22h
2batch_normalization_3/batchnorm/mul/ReadVariableOp2batch_normalization_3/batchnorm/mul/ReadVariableOp2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2b
/conv2d/kernel/Regularizer/Square/ReadVariableOp/conv2d/kernel/Regularizer/Square/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2f
1conv2d_1/kernel/Regularizer/Square/ReadVariableOp1conv2d_1/kernel/Regularizer/Square/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2X
*layer_normalization/Reshape/ReadVariableOp*layer_normalization/Reshape/ReadVariableOp2\
,layer_normalization/Reshape_1/ReadVariableOp,layer_normalization/Reshape_1/ReadVariableOp2R
'photoreceptor_rods_reike/ReadVariableOp'photoreceptor_rods_reike/ReadVariableOp2V
)photoreceptor_rods_reike/ReadVariableOp_1)photoreceptor_rods_reike/ReadVariableOp_12V
)photoreceptor_rods_reike/ReadVariableOp_2)photoreceptor_rods_reike/ReadVariableOp_22V
)photoreceptor_rods_reike/ReadVariableOp_3)photoreceptor_rods_reike/ReadVariableOp_32V
)photoreceptor_rods_reike/ReadVariableOp_4)photoreceptor_rods_reike/ReadVariableOp_42V
)photoreceptor_rods_reike/ReadVariableOp_5)photoreceptor_rods_reike/ReadVariableOp_52V
)photoreceptor_rods_reike/ReadVariableOp_6)photoreceptor_rods_reike/ReadVariableOp_62V
)photoreceptor_rods_reike/ReadVariableOp_7)photoreceptor_rods_reike/ReadVariableOp_72V
)photoreceptor_rods_reike/ReadVariableOp_8)photoreceptor_rods_reike/ReadVariableOp_82V
)photoreceptor_rods_reike/ReadVariableOp_9)photoreceptor_rods_reike/ReadVariableOp_92d
0photoreceptor_rods_reike/StatefulPartitionedCall0photoreceptor_rods_reike/StatefulPartitionedCall2Z
+photoreceptor_rods_reike/mul/ReadVariableOp+photoreceptor_rods_reike/mul/ReadVariableOp2^
-photoreceptor_rods_reike/mul_1/ReadVariableOp-photoreceptor_rods_reike/mul_1/ReadVariableOp2^
-photoreceptor_rods_reike/mul_2/ReadVariableOp-photoreceptor_rods_reike/mul_2/ReadVariableOp2^
-photoreceptor_rods_reike/mul_3/ReadVariableOp-photoreceptor_rods_reike/mul_3/ReadVariableOp2^
-photoreceptor_rods_reike/mul_4/ReadVariableOp-photoreceptor_rods_reike/mul_4/ReadVariableOp2^
-photoreceptor_rods_reike/mul_5/ReadVariableOp-photoreceptor_rods_reike/mul_5/ReadVariableOp2^
-photoreceptor_rods_reike/mul_6/ReadVariableOp-photoreceptor_rods_reike/mul_6/ReadVariableOp2^
-photoreceptor_rods_reike/mul_7/ReadVariableOp-photoreceptor_rods_reike/mul_7/ReadVariableOp2^
-photoreceptor_rods_reike/mul_8/ReadVariableOp-photoreceptor_rods_reike/mul_8/ReadVariableOp2^
-photoreceptor_rods_reike/mul_9/ReadVariableOp-photoreceptor_rods_reike/mul_9/ReadVariableOp:X T
0
_output_shapes
:         ┤'
 
_user_specified_nameinputs
╓
b
F__inference_activation_3_layer_call_and_return_conditional_losses_5169

inputs
identityN
SoftplusSoftplusinputs*
T0*'
_output_shapes
:         %^
IdentityIdentitySoftplus:activations:0*
T0*'
_output_shapes
:         %"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         %:O K
'
_output_shapes
:         %
 
_user_specified_nameinputs
▐8
Л
R__inference_photoreceptor_rods_reike_layer_call_and_return_conditional_losses_1809

inputs)
readvariableop_resource:-
mul_readvariableop_resource:+
readvariableop_1_resource:/
mul_1_readvariableop_resource:+
readvariableop_2_resource:/
mul_2_readvariableop_resource:+
readvariableop_3_resource:/
mul_3_readvariableop_resource:+
readvariableop_4_resource:/
mul_4_readvariableop_resource:+
readvariableop_5_resource:/
mul_5_readvariableop_resource:+
readvariableop_6_resource:/
mul_6_readvariableop_resource:+
readvariableop_7_resource:/
mul_7_readvariableop_resource:+
readvariableop_8_resource:/
mul_8_readvariableop_resource:+
readvariableop_9_resource:/
mul_9_readvariableop_resource:
unknown:
	unknown_0:
identityИвReadVariableOpвReadVariableOp_1вReadVariableOp_2вReadVariableOp_3вReadVariableOp_4вReadVariableOp_5вReadVariableOp_6вReadVariableOp_7вReadVariableOp_8вReadVariableOp_9вStatefulPartitionedCallвmul/ReadVariableOpвmul_1/ReadVariableOpвmul_2/ReadVariableOpвmul_3/ReadVariableOpвmul_4/ReadVariableOpвmul_5/ReadVariableOpвmul_6/ReadVariableOpвmul_7/ReadVariableOpвmul_8/ReadVariableOpвmul_9/ReadVariableOpf
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype0n
mul/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes

:*
dtype0g
mulMulReadVariableOp:value:0mul/ReadVariableOp:value:0*
T0*
_output_shapes

:j
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes

:*
dtype0r
mul_1/ReadVariableOpReadVariableOpmul_1_readvariableop_resource*
_output_shapes

:*
dtype0m
mul_1MulReadVariableOp_1:value:0mul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:j
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes

:*
dtype0r
mul_2/ReadVariableOpReadVariableOpmul_2_readvariableop_resource*
_output_shapes

:*
dtype0m
mul_2MulReadVariableOp_2:value:0mul_2/ReadVariableOp:value:0*
T0*
_output_shapes

:j
ReadVariableOp_3ReadVariableOpreadvariableop_3_resource*
_output_shapes

:*
dtype0r
mul_3/ReadVariableOpReadVariableOpmul_3_readvariableop_resource*
_output_shapes

:*
dtype0m
mul_3MulReadVariableOp_3:value:0mul_3/ReadVariableOp:value:0*
T0*
_output_shapes

:j
ReadVariableOp_4ReadVariableOpreadvariableop_4_resource*
_output_shapes

:*
dtype0r
mul_4/ReadVariableOpReadVariableOpmul_4_readvariableop_resource*
_output_shapes

:*
dtype0m
mul_4MulReadVariableOp_4:value:0mul_4/ReadVariableOp:value:0*
T0*
_output_shapes

:j
ReadVariableOp_5ReadVariableOpreadvariableop_5_resource*
_output_shapes

:*
dtype0r
mul_5/ReadVariableOpReadVariableOpmul_5_readvariableop_resource*
_output_shapes

:*
dtype0m
mul_5MulReadVariableOp_5:value:0mul_5/ReadVariableOp:value:0*
T0*
_output_shapes

:j
ReadVariableOp_6ReadVariableOpreadvariableop_6_resource*
_output_shapes

:*
dtype0r
mul_6/ReadVariableOpReadVariableOpmul_6_readvariableop_resource*
_output_shapes

:*
dtype0m
mul_6MulReadVariableOp_6:value:0mul_6/ReadVariableOp:value:0*
T0*
_output_shapes

:j
ReadVariableOp_7ReadVariableOpreadvariableop_7_resource*
_output_shapes

:*
dtype0r
mul_7/ReadVariableOpReadVariableOpmul_7_readvariableop_resource*
_output_shapes

:*
dtype0m
mul_7MulReadVariableOp_7:value:0mul_7/ReadVariableOp:value:0*
T0*
_output_shapes

:j
ReadVariableOp_8ReadVariableOpreadvariableop_8_resource*
_output_shapes

:*
dtype0r
mul_8/ReadVariableOpReadVariableOpmul_8_readvariableop_resource*
_output_shapes

:*
dtype0m
mul_8MulReadVariableOp_8:value:0mul_8/ReadVariableOp:value:0*
T0*
_output_shapes

:N
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   AZ
truedivRealDiv	mul_8:z:0truediv/y:output:0*
T0*
_output_shapes

:j
ReadVariableOp_9ReadVariableOpreadvariableop_9_resource*
_output_shapes

:*
dtype0r
mul_9/ReadVariableOpReadVariableOpmul_9_readvariableop_resource*
_output_shapes

:*
dtype0m
mul_9MulReadVariableOp_9:value:0mul_9/ReadVariableOp:value:0*
T0*
_output_shapes

:▒
StatefulPartitionedCallStatefulPartitionedCallinputsmul:z:0	mul_1:z:0	mul_2:z:0unknown	mul_3:z:0	unknown_0	mul_4:z:0	mul_5:z:0	mul_6:z:0	mul_7:z:0truediv:z:0	mul_9:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         ┤Т	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *$
fR
__inference_riekeModel_1187u
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*-
_output_shapes
:         ┤Т	А
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8^ReadVariableOp_9^StatefulPartitionedCall^mul/ReadVariableOp^mul_1/ReadVariableOp^mul_2/ReadVariableOp^mul_3/ReadVariableOp^mul_4/ReadVariableOp^mul_5/ReadVariableOp^mul_6/ReadVariableOp^mul_7/ReadVariableOp^mul_8/ReadVariableOp^mul_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:         ┤Т	: : : : : : : : : : : : : : : : : : : : : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_62$
ReadVariableOp_7ReadVariableOp_72$
ReadVariableOp_8ReadVariableOp_82$
ReadVariableOp_9ReadVariableOp_922
StatefulPartitionedCallStatefulPartitionedCall2(
mul/ReadVariableOpmul/ReadVariableOp2,
mul_1/ReadVariableOpmul_1/ReadVariableOp2,
mul_2/ReadVariableOpmul_2/ReadVariableOp2,
mul_3/ReadVariableOpmul_3/ReadVariableOp2,
mul_4/ReadVariableOpmul_4/ReadVariableOp2,
mul_5/ReadVariableOpmul_5/ReadVariableOp2,
mul_6/ReadVariableOpmul_6/ReadVariableOp2,
mul_7/ReadVariableOpmul_7/ReadVariableOp2,
mul_8/ReadVariableOpmul_8/ReadVariableOp2,
mul_9/ReadVariableOpmul_9/ReadVariableOp:U Q
-
_output_shapes
:         ┤Т	
 
_user_specified_nameinputs
╠
K
/__inference_gaussian_noise_2_layer_call_fn_5012

inputs
identity└
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_gaussian_noise_2_layer_call_and_return_conditional_losses_2108h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
├
_
C__inference_flatten_3_layer_call_and_return_conditional_losses_2123

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"    Z   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:         ZX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:         Z"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
┘
_
C__inference_reshape_1_layer_call_and_return_conditional_losses_1869

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskR
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :┤Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :'й
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:m
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:         ┤'a
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:         ┤'"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:         ┤Т	:U Q
-
_output_shapes
:         ┤Т	
 
_user_specified_nameinputs
▐8
Л
R__inference_photoreceptor_rods_reike_layer_call_and_return_conditional_losses_4440

inputs)
readvariableop_resource:-
mul_readvariableop_resource:+
readvariableop_1_resource:/
mul_1_readvariableop_resource:+
readvariableop_2_resource:/
mul_2_readvariableop_resource:+
readvariableop_3_resource:/
mul_3_readvariableop_resource:+
readvariableop_4_resource:/
mul_4_readvariableop_resource:+
readvariableop_5_resource:/
mul_5_readvariableop_resource:+
readvariableop_6_resource:/
mul_6_readvariableop_resource:+
readvariableop_7_resource:/
mul_7_readvariableop_resource:+
readvariableop_8_resource:/
mul_8_readvariableop_resource:+
readvariableop_9_resource:/
mul_9_readvariableop_resource:
unknown:
	unknown_0:
identityИвReadVariableOpвReadVariableOp_1вReadVariableOp_2вReadVariableOp_3вReadVariableOp_4вReadVariableOp_5вReadVariableOp_6вReadVariableOp_7вReadVariableOp_8вReadVariableOp_9вStatefulPartitionedCallвmul/ReadVariableOpвmul_1/ReadVariableOpвmul_2/ReadVariableOpвmul_3/ReadVariableOpвmul_4/ReadVariableOpвmul_5/ReadVariableOpвmul_6/ReadVariableOpвmul_7/ReadVariableOpвmul_8/ReadVariableOpвmul_9/ReadVariableOpf
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype0n
mul/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes

:*
dtype0g
mulMulReadVariableOp:value:0mul/ReadVariableOp:value:0*
T0*
_output_shapes

:j
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes

:*
dtype0r
mul_1/ReadVariableOpReadVariableOpmul_1_readvariableop_resource*
_output_shapes

:*
dtype0m
mul_1MulReadVariableOp_1:value:0mul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:j
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes

:*
dtype0r
mul_2/ReadVariableOpReadVariableOpmul_2_readvariableop_resource*
_output_shapes

:*
dtype0m
mul_2MulReadVariableOp_2:value:0mul_2/ReadVariableOp:value:0*
T0*
_output_shapes

:j
ReadVariableOp_3ReadVariableOpreadvariableop_3_resource*
_output_shapes

:*
dtype0r
mul_3/ReadVariableOpReadVariableOpmul_3_readvariableop_resource*
_output_shapes

:*
dtype0m
mul_3MulReadVariableOp_3:value:0mul_3/ReadVariableOp:value:0*
T0*
_output_shapes

:j
ReadVariableOp_4ReadVariableOpreadvariableop_4_resource*
_output_shapes

:*
dtype0r
mul_4/ReadVariableOpReadVariableOpmul_4_readvariableop_resource*
_output_shapes

:*
dtype0m
mul_4MulReadVariableOp_4:value:0mul_4/ReadVariableOp:value:0*
T0*
_output_shapes

:j
ReadVariableOp_5ReadVariableOpreadvariableop_5_resource*
_output_shapes

:*
dtype0r
mul_5/ReadVariableOpReadVariableOpmul_5_readvariableop_resource*
_output_shapes

:*
dtype0m
mul_5MulReadVariableOp_5:value:0mul_5/ReadVariableOp:value:0*
T0*
_output_shapes

:j
ReadVariableOp_6ReadVariableOpreadvariableop_6_resource*
_output_shapes

:*
dtype0r
mul_6/ReadVariableOpReadVariableOpmul_6_readvariableop_resource*
_output_shapes

:*
dtype0m
mul_6MulReadVariableOp_6:value:0mul_6/ReadVariableOp:value:0*
T0*
_output_shapes

:j
ReadVariableOp_7ReadVariableOpreadvariableop_7_resource*
_output_shapes

:*
dtype0r
mul_7/ReadVariableOpReadVariableOpmul_7_readvariableop_resource*
_output_shapes

:*
dtype0m
mul_7MulReadVariableOp_7:value:0mul_7/ReadVariableOp:value:0*
T0*
_output_shapes

:j
ReadVariableOp_8ReadVariableOpreadvariableop_8_resource*
_output_shapes

:*
dtype0r
mul_8/ReadVariableOpReadVariableOpmul_8_readvariableop_resource*
_output_shapes

:*
dtype0m
mul_8MulReadVariableOp_8:value:0mul_8/ReadVariableOp:value:0*
T0*
_output_shapes

:N
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   AZ
truedivRealDiv	mul_8:z:0truediv/y:output:0*
T0*
_output_shapes

:j
ReadVariableOp_9ReadVariableOpreadvariableop_9_resource*
_output_shapes

:*
dtype0r
mul_9/ReadVariableOpReadVariableOpmul_9_readvariableop_resource*
_output_shapes

:*
dtype0m
mul_9MulReadVariableOp_9:value:0mul_9/ReadVariableOp:value:0*
T0*
_output_shapes

:▒
StatefulPartitionedCallStatefulPartitionedCallinputsmul:z:0	mul_1:z:0	mul_2:z:0unknown	mul_3:z:0	unknown_0	mul_4:z:0	mul_5:z:0	mul_6:z:0	mul_7:z:0truediv:z:0	mul_9:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         ┤Т	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *$
fR
__inference_riekeModel_1187u
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*-
_output_shapes
:         ┤Т	А
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8^ReadVariableOp_9^StatefulPartitionedCall^mul/ReadVariableOp^mul_1/ReadVariableOp^mul_2/ReadVariableOp^mul_3/ReadVariableOp^mul_4/ReadVariableOp^mul_5/ReadVariableOp^mul_6/ReadVariableOp^mul_7/ReadVariableOp^mul_8/ReadVariableOp^mul_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:         ┤Т	: : : : : : : : : : : : : : : : : : : : : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_62$
ReadVariableOp_7ReadVariableOp_72$
ReadVariableOp_8ReadVariableOp_82$
ReadVariableOp_9ReadVariableOp_922
StatefulPartitionedCallStatefulPartitionedCall2(
mul/ReadVariableOpmul/ReadVariableOp2,
mul_1/ReadVariableOpmul_1/ReadVariableOp2,
mul_2/ReadVariableOpmul_2/ReadVariableOp2,
mul_3/ReadVariableOpmul_3/ReadVariableOp2,
mul_4/ReadVariableOpmul_4/ReadVariableOp2,
mul_5/ReadVariableOpmul_5/ReadVariableOp2,
mul_6/ReadVariableOpmul_6/ReadVariableOp2,
mul_7/ReadVariableOpmul_7/ReadVariableOp2,
mul_8/ReadVariableOpmul_8/ReadVariableOp2,
mul_9/ReadVariableOpmul_9/ReadVariableOp:U Q
-
_output_shapes
:         ┤Т	
 
_user_specified_nameinputs
ТQ
Ў	
while_body_1040
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_placeholder_4
while_placeholder_5
while_placeholder_6
while_range_1_limit_0
while_mul_sigma_0
while_strided_slice_x_fun_0
while_mul_3_gamma_0
while_add_3_eta_0
while_mul_4_phi_0
while_pow_cgmphill_01
while_readvariableop_resource_0:
while_mul_7_truediv_3_0
while_mul_8_beta_0"
while_truediv_1_hillaffinity_0
while_pow_1_hillcoef_0
while_truediv_2_mul_4_0
while_add_8_range_1_delta_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_identity_6
while_identity_7
while_identity_8
while_range_1_limit
while_mul_sigma
while_strided_slice_x_fun
while_mul_3_gamma
while_add_3_eta
while_mul_4_phi
while_pow_cgmphill/
while_readvariableop_resource:
while_mul_7_truediv_3
while_mul_8_beta 
while_truediv_1_hillaffinity
while_pow_1_hillcoef
while_truediv_2_mul_4
while_add_8_range_1_deltaИвwhile/ReadVariableOpP
while/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А┐b
	while/mulMulwhile/mul/x:output:0while_mul_sigma_0*
T0*
_output_shapes

:i
while/mul_1Mulwhile/mul:z:0while_placeholder_5*
T0*(
_output_shapes
:         Т	R
while/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *o<n
while/mul_2Mulwhile/mul_2/x:output:0while/mul_1:z:0*
T0*(
_output_shapes
:         Т	k
	while/addAddV2while_placeholder_5while/mul_2:z:0*
T0*(
_output_shapes
:         Т	M
while/sub/yConst*
_output_shapes
: *
dtype0*
value	B :Z
	while/subSubwhile_placeholderwhile/sub/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :\
while/add_1AddV2while/sub:z:0while/add_1/y:output:0*
T0*
_output_shapes
: ]
while/strided_slice/stack/0Const*
_output_shapes
: *
dtype0*
value	B : ]
while/strided_slice/stack/2Const*
_output_shapes
: *
dtype0*
value	B : к
while/strided_slice/stackPack$while/strided_slice/stack/0:output:0while/sub:z:0$while/strided_slice/stack/2:output:0*
N*
T0*
_output_shapes
:_
while/strided_slice/stack_1/0Const*
_output_shapes
: *
dtype0*
value	B : _
while/strided_slice/stack_1/2Const*
_output_shapes
: *
dtype0*
value	B : ▓
while/strided_slice/stack_1Pack&while/strided_slice/stack_1/0:output:0while/add_1:z:0&while/strided_slice/stack_1/2:output:0*
N*
T0*
_output_shapes
:p
while/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         к
while/strided_sliceStridedSlicewhile_strided_slice_x_fun_0"while/strided_slice/stack:output:0$while/strided_slice/stack_1:output:0$while/strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         Т	*

begin_mask*
end_mask*
shrink_axis_maskx
while/mul_3Mulwhile_mul_3_gamma_0while/strided_slice:output:0*
T0*(
_output_shapes
:         Т	g
while/add_2AddV2while/add:z:0while/mul_3:z:0*
T0*(
_output_shapes
:         Т	o
while/add_3AddV2while_placeholder_5while_add_3_eta_0*
T0*(
_output_shapes
:         Т	m
while/mul_4Mulwhile_mul_4_phi_0while_placeholder_4*
T0*(
_output_shapes
:         Т	g
while/sub_1Subwhile/add_3:z:0while/mul_4:z:0*
T0*(
_output_shapes
:         Т	R
while/mul_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *o<n
while/mul_5Mulwhile/mul_5/x:output:0while/sub_1:z:0*
T0*(
_output_shapes
:         Т	m
while/add_4AddV2while_placeholder_4while/mul_5:z:0*
T0*(
_output_shapes
:         Т	n
	while/powPowwhile_placeholder_3while_pow_cgmphill_0*
T0*(
_output_shapes
:         Т	t
while/ReadVariableOpReadVariableOpwhile_readvariableop_resource_0*
_output_shapes

:*
dtype0r
while/mul_6Mulwhile/ReadVariableOp:value:0while/pow:z:0*
T0*(
_output_shapes
:         Т	o
while/mul_7Mulwhile_mul_7_truediv_3_0while/mul_6:z:0*
T0*(
_output_shapes
:         Т	T
while/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @v
while/truedivRealDivwhile/mul_7:z:0while/truediv/y:output:0*
T0*(
_output_shapes
:         Т	n
while/mul_8Mulwhile_mul_8_beta_0while_placeholder_2*
T0*(
_output_shapes
:         Т	i
while/sub_2Subwhile/truediv:z:0while/mul_8:z:0*
T0*(
_output_shapes
:         Т	R
while/mul_9/xConst*
_output_shapes
: *
dtype0*
valueB
 *o<n
while/mul_9Mulwhile/mul_9/x:output:0while/sub_2:z:0*
T0*(
_output_shapes
:         Т	m
while/add_5AddV2while_placeholder_2while/mul_9:z:0*
T0*(
_output_shapes
:         Т	~
while/truediv_1RealDivwhile/add_5:z:0while_truediv_1_hillaffinity_0*
T0*(
_output_shapes
:         Т	r
while/pow_1Powwhile/truediv_1:z:0while_pow_1_hillcoef_0*
T0*(
_output_shapes
:         Т	R
while/add_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?p
while/add_6AddV2while/add_6/x:output:0while/pow_1:z:0*
T0*(
_output_shapes
:         Т	w
while/truediv_2RealDivwhile_truediv_2_mul_4_0while/add_6:z:0*
T0*(
_output_shapes
:         Т	p
while/mul_10Mulwhile_placeholder_4while_placeholder_3*
T0*(
_output_shapes
:         Т	l
while/sub_3Subwhile_placeholder_6while/mul_10:z:0*
T0*(
_output_shapes
:         Т	S
while/mul_11/xConst*
_output_shapes
: *
dtype0*
valueB
 *o<p
while/mul_11Mulwhile/mul_11/x:output:0while/sub_3:z:0*
T0*(
_output_shapes
:         Т	n
while/add_7AddV2while_placeholder_3while/mul_11:z:0*
T0*(
_output_shapes
:         Т	╕
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/add_7:z:0*
_output_shapes
: *
element_dtype0:щш╥e
while/add_8AddV2while_placeholderwhile_add_8_range_1_delta_0*
T0*
_output_shapes
: O
while/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_9AddV2while_while_loop_counterwhile/add_9/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_9:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: [
while/Identity_2Identitywhile/add_8:z:0^while/NoOp*
T0*
_output_shapes
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: m
while/Identity_4Identitywhile/add_5:z:0^while/NoOp*
T0*(
_output_shapes
:         Т	m
while/Identity_5Identitywhile/add_7:z:0^while/NoOp*
T0*(
_output_shapes
:         Т	m
while/Identity_6Identitywhile/add_4:z:0^while/NoOp*
T0*(
_output_shapes
:         Т	m
while/Identity_7Identitywhile/add_2:z:0^while/NoOp*
T0*(
_output_shapes
:         Т	q
while/Identity_8Identitywhile/truediv_2:z:0^while/NoOp*
T0*(
_output_shapes
:         Т	c

while/NoOpNoOp^while/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "$
while_add_3_etawhile_add_3_eta_0"8
while_add_8_range_1_deltawhile_add_8_range_1_delta_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"-
while_identity_6while/Identity_6:output:0"-
while_identity_7while/Identity_7:output:0"-
while_identity_8while/Identity_8:output:0"(
while_mul_3_gammawhile_mul_3_gamma_0"$
while_mul_4_phiwhile_mul_4_phi_0"0
while_mul_7_truediv_3while_mul_7_truediv_3_0"&
while_mul_8_betawhile_mul_8_beta_0"$
while_mul_sigmawhile_mul_sigma_0".
while_pow_1_hillcoefwhile_pow_1_hillcoef_0"*
while_pow_cgmphillwhile_pow_cgmphill_0",
while_range_1_limitwhile_range_1_limit_0"@
while_readvariableop_resourcewhile_readvariableop_resource_0"8
while_strided_slice_x_funwhile_strided_slice_x_fun_0">
while_truediv_1_hillaffinitywhile_truediv_1_hillaffinity_0"0
while_truediv_2_mul_4while_truediv_2_mul_4_0*(
_construction_contextkEagerRuntime*Д
_input_shapesЄ
я: : : : :         Т	:         Т	:         Т	:         Т	:         Т	: ::         ┤Т	::::: :::::: 2,
while/ReadVariableOpwhile/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         Т	:.*
(
_output_shapes
:         Т	:.*
(
_output_shapes
:         Т	:.*
(
_output_shapes
:         Т	:.*
(
_output_shapes
:         Т	:	

_output_shapes
: :$
 

_output_shapes

::3/
-
_output_shapes
:         ┤Т	:$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::

_output_shapes
: 
░
D
(__inference_reshape_3_layer_call_fn_4817

inputs
identity╣
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_reshape_3_layer_call_and_return_conditional_losses_2034h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ╨:P L
(
_output_shapes
:         ╨
 
_user_specified_nameinputs
▐
Р
M__inference_layer_normalization_layer_call_and_return_conditional_losses_4494

inputs5
reshape_readvariableop_resource:x'7
!reshape_1_readvariableop_resource:x'
identityИвReshape/ReadVariableOpвReshape_1/ReadVariableOps
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         Р
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*/
_output_shapes
:         *
	keep_dims(u
moments/StopGradientStopGradientmoments/mean:output:0*
T0*/
_output_shapes
:         П
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*/
_output_shapes
:         x'w
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         п
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*/
_output_shapes
:         *
	keep_dims(z
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*"
_output_shapes
:x'*
dtype0f
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   x      '   {
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*&
_output_shapes
:x'~
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*"
_output_shapes
:x'*
dtype0h
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   x      '   Б
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:x'T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Х┐╓3Е
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*/
_output_shapes
:         e
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*/
_output_shapes
:         u
batchnorm/mulMulbatchnorm/Rsqrt:y:0Reshape:output:0*
T0*/
_output_shapes
:         x'k
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*/
_output_shapes
:         x'z
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*/
_output_shapes
:         x'w
batchnorm/subSubReshape_1:output:0batchnorm/mul_2:z:0*
T0*/
_output_shapes
:         x'z
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*/
_output_shapes
:         x'j
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*/
_output_shapes
:         x'z
NoOpNoOp^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         x': : 20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:W S
/
_output_shapes
:         x'
 
_user_specified_nameinputs
Ю
│
D__inference_CNNs_start_layer_call_and_return_conditional_losses_4525

inputs8
conv2d_readvariableop_resource:		x-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpв3CNNs_start/kernel/Regularizer/Square/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:		x*
dtype0▒
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
data_formatNCHW*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ф
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
data_formatNCHWЪ
3CNNs_start/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:		x*
dtype0Ь
$CNNs_start/kernel/Regularizer/SquareSquare;CNNs_start/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		x|
#CNNs_start/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             б
!CNNs_start/kernel/Regularizer/SumSum(CNNs_start/kernel/Regularizer/Square:y:0,CNNs_start/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#CNNs_start/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:г
!CNNs_start/kernel/Regularizer/mulMul,CNNs_start/kernel/Regularizer/mul/x:output:0*CNNs_start/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:         н
NoOpNoOp^BiasAdd/ReadVariableOp4^CNNs_start/kernel/Regularizer/Square/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         x': : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2j
3CNNs_start/kernel/Regularizer/Square/ReadVariableOp3CNNs_start/kernel/Regularizer/Square/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         x'
 
_user_specified_nameinputs
ж	
i
J__inference_gaussian_noise_1_layer_call_and_return_conditional_losses_4856

inputs
identityИ;
ShapeShapeinputs*
T0*
_output_shapes
:W
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=С
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*/
_output_shapes
:         	*
dtype0Ю
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*/
_output_shapes
:         	Д
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*/
_output_shapes
:         	a
addAddV2inputsrandom_normal:z:0*
T0*/
_output_shapes
:         	W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:         	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         	:W S
/
_output_shapes
:         	
 
_user_specified_nameinputs
┼
_
C__inference_flatten_1_layer_call_and_return_conditional_losses_2009

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"    ╨  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         ╨Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         ╨"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         	:W S
/
_output_shapes
:         	
 
_user_specified_nameinputs
щ
Ь
'__inference_conv2d_1_layer_call_fn_4881

inputs!
unknown:
	unknown_0:
identityИвStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_2065w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         	: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         	
 
_user_specified_nameinputs
∙
╗
__inference_loss_fn_0_5180V
<cnns_start_kernel_regularizer_square_readvariableop_resource:		x
identityИв3CNNs_start/kernel/Regularizer/Square/ReadVariableOp╕
3CNNs_start/kernel/Regularizer/Square/ReadVariableOpReadVariableOp<cnns_start_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:		x*
dtype0Ь
$CNNs_start/kernel/Regularizer/SquareSquare;CNNs_start/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		x|
#CNNs_start/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             б
!CNNs_start/kernel/Regularizer/SumSum(CNNs_start/kernel/Regularizer/Square:y:0,CNNs_start/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#CNNs_start/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:г
!CNNs_start/kernel/Regularizer/mulMul,CNNs_start/kernel/Regularizer/mul/x:output:0*CNNs_start/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: c
IdentityIdentity%CNNs_start/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: |
NoOpNoOp4^CNNs_start/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2j
3CNNs_start/kernel/Regularizer/Square/ReadVariableOp3CNNs_start/kernel/Regularizer/Square/ReadVariableOp
д	
g
H__inference_gaussian_noise_layer_call_and_return_conditional_losses_4680

inputs
identityИ;
ShapeShapeinputs*
T0*
_output_shapes
:W
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=С
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*/
_output_shapes
:         *
dtype0Ю
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*/
_output_shapes
:         Д
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*/
_output_shapes
:         a
addAddV2inputsrandom_normal:z:0*
T0*/
_output_shapes
:         W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
┼
_
C__inference_flatten_1_layer_call_and_return_conditional_losses_4732

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"    ╨  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         ╨Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         ╨"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         	:W S
/
_output_shapes
:         	
 
_user_specified_nameinputs
╒
╖
__inference_loss_fn_2_5202T
:conv2d_1_kernel_regularizer_square_readvariableop_resource:
identityИв1conv2d_1/kernel/Regularizer/Square/ReadVariableOp┤
1conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:conv2d_1_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:*
dtype0Ш
"conv2d_1/kernel/Regularizer/SquareSquare9conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
!conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             Ы
conv2d_1/kernel/Regularizer/SumSum&conv2d_1/kernel/Regularizer/Square:y:0*conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:Э
conv2d_1/kernel/Regularizer/mulMul*conv2d_1/kernel/Regularizer/mul/x:output:0(conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentity#conv2d_1/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: z
NoOpNoOp2^conv2d_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1conv2d_1/kernel/Regularizer/Square/ReadVariableOp1conv2d_1/kernel/Regularizer/Square/ReadVariableOp
о
D
(__inference_flatten_2_layer_call_fn_4902

inputs
identity▒
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         Z* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_flatten_2_layer_call_and_return_conditional_losses_2077`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         Z"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
л%
ъ
M__inference_batch_normalization_layer_call_and_return_conditional_losses_1444

inputs6
'assignmovingavg_readvariableop_resource:	╨*8
)assignmovingavg_1_readvariableop_resource:	╨*4
%batchnorm_mul_readvariableop_resource:	╨*0
!batchnorm_readvariableop_resource:	╨*
identityИвAssignMovingAvgвAssignMovingAvg/ReadVariableOpвAssignMovingAvg_1в AssignMovingAvg_1/ReadVariableOpвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: А
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	╨**
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	╨*И
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:         ╨*l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Я
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	╨**
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:╨**
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:╨**
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<Г
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:╨**
dtype0В
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:╨*y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:╨*м
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<З
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:╨**
dtype0И
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:╨*
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:╨*┤
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:╨*Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:╨*
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:╨**
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:╨*d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         ╨*i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:╨*w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:╨**
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:╨*s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         ╨*c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:         ╨*ъ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ╨*: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:         ╨*
 
_user_specified_nameinputs
└
E
)__inference_activation_layer_call_fn_4685

inputs
identity║
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_activation_layer_call_and_return_conditional_losses_1979h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
╠
о
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_5099

inputs/
!batchnorm_readvariableop_resource:Z3
%batchnorm_mul_readvariableop_resource:Z1
#batchnorm_readvariableop_1_resource:Z1
#batchnorm_readvariableop_2_resource:Z
identityИвbatchnorm/ReadVariableOpвbatchnorm/ReadVariableOp_1вbatchnorm/ReadVariableOp_2вbatchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:Z*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:ZP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:Z~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:Z*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Zc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:         Zz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:Z*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:Zz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:Z*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Zr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:         Zb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:         Z║
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         Z: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:         Z
 
_user_specified_nameinputs
─
G
+__inference_activation_1_layer_call_fn_4861

inputs
identity╝
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_activation_1_layer_call_and_return_conditional_losses_2047h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         	:W S
/
_output_shapes
:         	
 
_user_specified_nameinputs
├
_
C__inference_flatten_3_layer_call_and_return_conditional_losses_5053

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"    Z   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:         ZX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:         Z"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
го
н
I__inference_PRFR_CNN2D_RODS_layer_call_and_return_conditional_losses_2197

inputs/
photoreceptor_rods_reike_1810:/
photoreceptor_rods_reike_1812:/
photoreceptor_rods_reike_1814:/
photoreceptor_rods_reike_1816:/
photoreceptor_rods_reike_1818:/
photoreceptor_rods_reike_1820:/
photoreceptor_rods_reike_1822:/
photoreceptor_rods_reike_1824:/
photoreceptor_rods_reike_1826:/
photoreceptor_rods_reike_1828:/
photoreceptor_rods_reike_1830:/
photoreceptor_rods_reike_1832:/
photoreceptor_rods_reike_1834:/
photoreceptor_rods_reike_1836:/
photoreceptor_rods_reike_1838:/
photoreceptor_rods_reike_1840:/
photoreceptor_rods_reike_1842:/
photoreceptor_rods_reike_1844:/
photoreceptor_rods_reike_1846:/
photoreceptor_rods_reike_1848:/
photoreceptor_rods_reike_1850:/
photoreceptor_rods_reike_1852:.
layer_normalization_1902:x'.
layer_normalization_1904:x')
cnns_start_1924:		x
cnns_start_1926:'
batch_normalization_1937:	╨*'
batch_normalization_1939:	╨*'
batch_normalization_1941:	╨*'
batch_normalization_1943:	╨*%
conv2d_1998:
conv2d_2000:)
batch_normalization_1_2011:	╨)
batch_normalization_1_2013:	╨)
batch_normalization_1_2015:	╨)
batch_normalization_1_2017:	╨'
conv2d_1_2066:
conv2d_1_2068:(
batch_normalization_2_2079:Z(
batch_normalization_2_2081:Z(
batch_normalization_2_2083:Z(
batch_normalization_2_2085:Z(
batch_normalization_3_2125:Z(
batch_normalization_3_2127:Z(
batch_normalization_3_2129:Z(
batch_normalization_3_2131:Z

dense_2151:Z%

dense_2153:%
identity

identity_1Ив"CNNs_start/StatefulPartitionedCallв3CNNs_start/kernel/Regularizer/Square/ReadVariableOpв+batch_normalization/StatefulPartitionedCallв-batch_normalization_1/StatefulPartitionedCallв-batch_normalization_2/StatefulPartitionedCallв-batch_normalization_3/StatefulPartitionedCallвconv2d/StatefulPartitionedCallв/conv2d/kernel/Regularizer/Square/ReadVariableOpв conv2d_1/StatefulPartitionedCallв1conv2d_1/kernel/Regularizer/Square/ReadVariableOpвdense/StatefulPartitionedCallв.dense/kernel/Regularizer/Square/ReadVariableOpв+layer_normalization/StatefulPartitionedCallв0photoreceptor_rods_reike/StatefulPartitionedCall╜
reshape/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         ┤Т	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_reshape_layer_call_and_return_conditional_losses_1746с
0photoreceptor_rods_reike/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0photoreceptor_rods_reike_1810photoreceptor_rods_reike_1812photoreceptor_rods_reike_1814photoreceptor_rods_reike_1816photoreceptor_rods_reike_1818photoreceptor_rods_reike_1820photoreceptor_rods_reike_1822photoreceptor_rods_reike_1824photoreceptor_rods_reike_1826photoreceptor_rods_reike_1828photoreceptor_rods_reike_1830photoreceptor_rods_reike_1832photoreceptor_rods_reike_1834photoreceptor_rods_reike_1836photoreceptor_rods_reike_1838photoreceptor_rods_reike_1840photoreceptor_rods_reike_1842photoreceptor_rods_reike_1844photoreceptor_rods_reike_1846photoreceptor_rods_reike_1848photoreceptor_rods_reike_1850photoreceptor_rods_reike_1852*"
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         ┤Т	*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_photoreceptor_rods_reike_layer_call_and_return_conditional_losses_1809ў
reshape_1/PartitionedCallPartitionedCall9photoreceptor_rods_reike/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ┤'* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_reshape_1_layer_call_and_return_conditional_losses_1869Е
,tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"    <           З
.tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                З
.tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            ь
&tf.__operators__.getitem/strided_sliceStridedSlice"reshape_1/PartitionedCall:output:05tf.__operators__.getitem/strided_slice/stack:output:07tf.__operators__.getitem/strided_slice/stack_1:output:07tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         x'*

begin_mask*
end_mask╩
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCall/tf.__operators__.getitem/strided_slice:output:0layer_normalization_1902layer_normalization_1904*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         x'*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_layer_normalization_layer_call_and_return_conditional_losses_1901л
"CNNs_start/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0cnns_start_1924cnns_start_1926*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_CNNs_start_layer_call_and_return_conditional_losses_1923▌
flatten/PartitionedCallPartitionedCall+CNNs_start/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ╨** 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_1935ь
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0batch_normalization_1937batch_normalization_1939batch_normalization_1941batch_normalization_1943*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ╨**&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_1397ё
reshape_2/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_reshape_2_layer_call_and_return_conditional_losses_1960ч
max_pooling2d/PartitionedCallPartitionedCall"reshape_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_1966э
gaussian_noise/PartitionedCallPartitionedCall&max_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_gaussian_noise_layer_call_and_return_conditional_losses_1972ц
activation/PartitionedCallPartitionedCall'gaussian_noise/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_activation_layer_call_and_return_conditional_losses_1979К
conv2d/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0conv2d_1998conv2d_2000*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_1997▌
flatten_1/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ╨* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_flatten_1_layer_call_and_return_conditional_losses_2009·
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0batch_normalization_1_2011batch_normalization_1_2013batch_normalization_1_2015batch_normalization_1_2017*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ╨*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1491є
reshape_3/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_reshape_3_layer_call_and_return_conditional_losses_2034э
 gaussian_noise_1/PartitionedCallPartitionedCall"reshape_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_gaussian_noise_1_layer_call_and_return_conditional_losses_2040ь
activation_1/PartitionedCallPartitionedCall)gaussian_noise_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_activation_1_layer_call_and_return_conditional_losses_2047Ф
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0conv2d_1_2066conv2d_1_2068*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_2065▐
flatten_2/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         Z* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_flatten_2_layer_call_and_return_conditional_losses_2077∙
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0batch_normalization_2_2079batch_normalization_2_2081batch_normalization_2_2083batch_normalization_2_2085*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         Z*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1573є
reshape_4/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_reshape_4_layer_call_and_return_conditional_losses_2102э
 gaussian_noise_2/PartitionedCallPartitionedCall"reshape_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_gaussian_noise_2_layer_call_and_return_conditional_losses_2108ь
activation_2/PartitionedCallPartitionedCall)gaussian_noise_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_activation_2_layer_call_and_return_conditional_losses_2115┌
flatten_3/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         Z* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_flatten_3_layer_call_and_return_conditional_losses_2123∙
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0batch_normalization_3_2125batch_normalization_3_2127batch_normalization_3_2129batch_normalization_3_2131*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         Z*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1655С
dense/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0
dense_2151
dense_2153*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         %*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_2150┬
)dense/ActivityRegularizer/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *4
f/R-
+__inference_dense_activity_regularizer_1726u
dense/ActivityRegularizer/ShapeShape&dense/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:w
-dense/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/dense/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/dense/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╙
'dense/ActivityRegularizer/strided_sliceStridedSlice(dense/ActivityRegularizer/Shape:output:06dense/ActivityRegularizer/strided_slice/stack:output:08dense/ActivityRegularizer/strided_slice/stack_1:output:08dense/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskИ
dense/ActivityRegularizer/CastCast0dense/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: е
!dense/ActivityRegularizer/truedivRealDiv2dense/ActivityRegularizer/PartitionedCall:output:0"dense/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: с
activation_3/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         %* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_activation_3_layer_call_and_return_conditional_losses_2169Л
3CNNs_start/kernel/Regularizer/Square/ReadVariableOpReadVariableOpcnns_start_1924*&
_output_shapes
:		x*
dtype0Ь
$CNNs_start/kernel/Regularizer/SquareSquare;CNNs_start/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		x|
#CNNs_start/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             б
!CNNs_start/kernel/Regularizer/SumSum(CNNs_start/kernel/Regularizer/Square:y:0,CNNs_start/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#CNNs_start/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:г
!CNNs_start/kernel/Regularizer/mulMul,CNNs_start/kernel/Regularizer/mul/x:output:0*CNNs_start/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: Г
/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_1998*&
_output_shapes
:*
dtype0Ф
 conv2d/kernel/Regularizer/SquareSquare7conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:x
conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             Х
conv2d/kernel/Regularizer/SumSum$conv2d/kernel/Regularizer/Square:y:0(conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:Ч
conv2d/kernel/Regularizer/mulMul(conv2d/kernel/Regularizer/mul/x:output:0&conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: З
1conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_1_2066*&
_output_shapes
:*
dtype0Ш
"conv2d_1/kernel/Regularizer/SquareSquare9conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
!conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             Ы
conv2d_1/kernel/Regularizer/SumSum&conv2d_1/kernel/Regularizer/Square:y:0*conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:Э
conv2d_1/kernel/Regularizer/mulMul*conv2d_1/kernel/Regularizer/mul/x:output:0(conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: y
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp
dense_2151*
_output_shapes

:Z%*
dtype0К
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:Z%o
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Т
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:Ф
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: t
IdentityIdentity%activation_3/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         %e

Identity_1Identity%dense/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: ╗
NoOpNoOp#^CNNs_start/StatefulPartitionedCall4^CNNs_start/kernel/Regularizer/Square/ReadVariableOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall^conv2d/StatefulPartitionedCall0^conv2d/kernel/Regularizer/Square/ReadVariableOp!^conv2d_1/StatefulPartitionedCall2^conv2d_1/kernel/Regularizer/Square/ReadVariableOp^dense/StatefulPartitionedCall/^dense/kernel/Regularizer/Square/ReadVariableOp,^layer_normalization/StatefulPartitionedCall1^photoreceptor_rods_reike/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*П
_input_shapes~
|:         ┤': : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
"CNNs_start/StatefulPartitionedCall"CNNs_start/StatefulPartitionedCall2j
3CNNs_start/kernel/Regularizer/Square/ReadVariableOp3CNNs_start/kernel/Regularizer/Square/ReadVariableOp2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2b
/conv2d/kernel/Regularizer/Square/ReadVariableOp/conv2d/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2f
1conv2d_1/kernel/Regularizer/Square/ReadVariableOp1conv2d_1/kernel/Regularizer/Square/ReadVariableOp2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2d
0photoreceptor_rods_reike/StatefulPartitionedCall0photoreceptor_rods_reike/StatefulPartitionedCall:X T
0
_output_shapes
:         ┤'
 
_user_specified_nameinputs
Р%
ш
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1702

inputs5
'assignmovingavg_readvariableop_resource:Z7
)assignmovingavg_1_readvariableop_resource:Z3
%batchnorm_mul_readvariableop_resource:Z/
!batchnorm_readvariableop_resource:Z
identityИвAssignMovingAvgвAssignMovingAvg/ReadVariableOpвAssignMovingAvg_1в AssignMovingAvg_1/ReadVariableOpвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:Z*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:ZЗ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:         Zl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ю
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:Z*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:Z*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:Z*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<В
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:Z*
dtype0Б
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:Zx
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:Zм
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<Ж
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:Z*
dtype0З
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:Z~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:Z┤
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:ZP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:Z~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:Z*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Zc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:         Zh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:Zv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:Z*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Zr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:         Zb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:         Zъ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         Z: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:         Z
 
_user_specified_nameinputs
л
╤
2__inference_batch_normalization_layer_call_fn_4549

inputs
unknown:	╨*
	unknown_0:	╨*
	unknown_1:	╨*
	unknown_2:	╨*
identityИвStatefulPartitionedCallА
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ╨**&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_1397p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ╨*`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ╨*: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ╨*
 
_user_specified_nameinputs
Т
f
J__inference_gaussian_noise_1_layer_call_and_return_conditional_losses_4845

inputs
identityV
IdentityIdentityinputs*
T0*/
_output_shapes
:         	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         	:W S
/
_output_shapes
:         	
 
_user_specified_nameinputs
л%
ъ
M__inference_batch_normalization_layer_call_and_return_conditional_losses_4616

inputs6
'assignmovingavg_readvariableop_resource:	╨*8
)assignmovingavg_1_readvariableop_resource:	╨*4
%batchnorm_mul_readvariableop_resource:	╨*0
!batchnorm_readvariableop_resource:	╨*
identityИвAssignMovingAvgвAssignMovingAvg/ReadVariableOpвAssignMovingAvg_1в AssignMovingAvg_1/ReadVariableOpвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: А
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	╨**
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	╨*И
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:         ╨*l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Я
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	╨**
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:╨**
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:╨**
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<Г
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:╨**
dtype0В
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:╨*y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:╨*м
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<З
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:╨**
dtype0И
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:╨*
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:╨*┤
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:╨*Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:╨*
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:╨**
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:╨*d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         ╨*i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:╨*w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:╨**
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:╨*s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         ╨*c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:         ╨*ъ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ╨*: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:         ╨*
 
_user_specified_nameinputs
Р
d
H__inference_gaussian_noise_layer_call_and_return_conditional_losses_1972

inputs
identityV
IdentityIdentityinputs*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
ы

]
A__inference_reshape_layer_call_and_return_conditional_losses_1746

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskR
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :┤R
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :Т	П
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:j
ReshapeReshapeinputsReshape/shape:output:0*
T0*-
_output_shapes
:         ┤Т	^
IdentityIdentityReshape:output:0*
T0*-
_output_shapes
:         ┤Т	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ┤':X T
0
_output_shapes
:         ┤'
 
_user_specified_nameinputs
ж
б
?__inference_dense_layer_call_and_return_conditional_losses_2150

inputs0
matmul_readvariableop_resource:Z%-
biasadd_readvariableop_resource:%
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpв.dense/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:Z%*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         %r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:%*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         %Н
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:Z%*
dtype0К
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:Z%o
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Т
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:Ф
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         %и
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         Z: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:         Z
 
_user_specified_nameinputs
Ю
h
/__inference_gaussian_noise_2_layer_call_fn_5017

inputs
identityИвStatefulPartitionedCall╨
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_gaussian_noise_2_layer_call_and_return_conditional_losses_2356w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
╠
_
C__inference_reshape_2_layer_call_and_return_conditional_losses_4635

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :й
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:         `
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ╨*:P L
(
_output_shapes
:         ╨*
 
_user_specified_nameinputs
▄
░
M__inference_batch_normalization_layer_call_and_return_conditional_losses_4582

inputs0
!batchnorm_readvariableop_resource:	╨*4
%batchnorm_mul_readvariableop_resource:	╨*2
#batchnorm_readvariableop_1_resource:	╨*2
#batchnorm_readvariableop_2_resource:	╨*
identityИвbatchnorm/ReadVariableOpвbatchnorm/ReadVariableOp_1вbatchnorm/ReadVariableOp_2вbatchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:╨**
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:╨*Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:╨*
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:╨**
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:╨*d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         ╨*{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:╨**
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:╨*{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:╨**
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:╨*s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         ╨*c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:         ╨*║
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ╨*: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:         ╨*
 
_user_specified_nameinputs
Р
d
H__inference_gaussian_noise_layer_call_and_return_conditional_losses_4669

inputs
identityV
IdentityIdentityinputs*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
╢│
н
I__inference_PRFR_CNN2D_RODS_layer_call_and_return_conditional_losses_3356
input_1/
photoreceptor_rods_reike_3196:/
photoreceptor_rods_reike_3198:/
photoreceptor_rods_reike_3200:/
photoreceptor_rods_reike_3202:/
photoreceptor_rods_reike_3204:/
photoreceptor_rods_reike_3206:/
photoreceptor_rods_reike_3208:/
photoreceptor_rods_reike_3210:/
photoreceptor_rods_reike_3212:/
photoreceptor_rods_reike_3214:/
photoreceptor_rods_reike_3216:/
photoreceptor_rods_reike_3218:/
photoreceptor_rods_reike_3220:/
photoreceptor_rods_reike_3222:/
photoreceptor_rods_reike_3224:/
photoreceptor_rods_reike_3226:/
photoreceptor_rods_reike_3228:/
photoreceptor_rods_reike_3230:/
photoreceptor_rods_reike_3232:/
photoreceptor_rods_reike_3234:/
photoreceptor_rods_reike_3236:/
photoreceptor_rods_reike_3238:.
layer_normalization_3246:x'.
layer_normalization_3248:x')
cnns_start_3251:		x
cnns_start_3253:'
batch_normalization_3257:	╨*'
batch_normalization_3259:	╨*'
batch_normalization_3261:	╨*'
batch_normalization_3263:	╨*%
conv2d_3270:
conv2d_3272:)
batch_normalization_1_3276:	╨)
batch_normalization_1_3278:	╨)
batch_normalization_1_3280:	╨)
batch_normalization_1_3282:	╨'
conv2d_1_3288:
conv2d_1_3290:(
batch_normalization_2_3294:Z(
batch_normalization_2_3296:Z(
batch_normalization_2_3298:Z(
batch_normalization_2_3300:Z(
batch_normalization_3_3307:Z(
batch_normalization_3_3309:Z(
batch_normalization_3_3311:Z(
batch_normalization_3_3313:Z

dense_3316:Z%

dense_3318:%
identity

identity_1Ив"CNNs_start/StatefulPartitionedCallв3CNNs_start/kernel/Regularizer/Square/ReadVariableOpв+batch_normalization/StatefulPartitionedCallв-batch_normalization_1/StatefulPartitionedCallв-batch_normalization_2/StatefulPartitionedCallв-batch_normalization_3/StatefulPartitionedCallвconv2d/StatefulPartitionedCallв/conv2d/kernel/Regularizer/Square/ReadVariableOpв conv2d_1/StatefulPartitionedCallв1conv2d_1/kernel/Regularizer/Square/ReadVariableOpвdense/StatefulPartitionedCallв.dense/kernel/Regularizer/Square/ReadVariableOpв&gaussian_noise/StatefulPartitionedCallв(gaussian_noise_1/StatefulPartitionedCallв(gaussian_noise_2/StatefulPartitionedCallв+layer_normalization/StatefulPartitionedCallв0photoreceptor_rods_reike/StatefulPartitionedCall╛
reshape/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         ┤Т	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_reshape_layer_call_and_return_conditional_losses_1746с
0photoreceptor_rods_reike/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0photoreceptor_rods_reike_3196photoreceptor_rods_reike_3198photoreceptor_rods_reike_3200photoreceptor_rods_reike_3202photoreceptor_rods_reike_3204photoreceptor_rods_reike_3206photoreceptor_rods_reike_3208photoreceptor_rods_reike_3210photoreceptor_rods_reike_3212photoreceptor_rods_reike_3214photoreceptor_rods_reike_3216photoreceptor_rods_reike_3218photoreceptor_rods_reike_3220photoreceptor_rods_reike_3222photoreceptor_rods_reike_3224photoreceptor_rods_reike_3226photoreceptor_rods_reike_3228photoreceptor_rods_reike_3230photoreceptor_rods_reike_3232photoreceptor_rods_reike_3234photoreceptor_rods_reike_3236photoreceptor_rods_reike_3238*"
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         ┤Т	*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_photoreceptor_rods_reike_layer_call_and_return_conditional_losses_1809ў
reshape_1/PartitionedCallPartitionedCall9photoreceptor_rods_reike/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ┤'* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_reshape_1_layer_call_and_return_conditional_losses_1869Е
,tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"    <           З
.tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                З
.tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            ь
&tf.__operators__.getitem/strided_sliceStridedSlice"reshape_1/PartitionedCall:output:05tf.__operators__.getitem/strided_slice/stack:output:07tf.__operators__.getitem/strided_slice/stack_1:output:07tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         x'*

begin_mask*
end_mask╩
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCall/tf.__operators__.getitem/strided_slice:output:0layer_normalization_3246layer_normalization_3248*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         x'*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_layer_normalization_layer_call_and_return_conditional_losses_1901л
"CNNs_start/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0cnns_start_3251cnns_start_3253*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_CNNs_start_layer_call_and_return_conditional_losses_1923▌
flatten/PartitionedCallPartitionedCall+CNNs_start/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ╨** 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_1935ъ
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0batch_normalization_3257batch_normalization_3259batch_normalization_3261batch_normalization_3263*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ╨**$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_1444ё
reshape_2/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_reshape_2_layer_call_and_return_conditional_losses_1960ч
max_pooling2d/PartitionedCallPartitionedCall"reshape_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_1966¤
&gaussian_noise/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_gaussian_noise_layer_call_and_return_conditional_losses_2456ю
activation/PartitionedCallPartitionedCall/gaussian_noise/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_activation_layer_call_and_return_conditional_losses_1979К
conv2d/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0conv2d_3270conv2d_3272*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_1997▌
flatten_1/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ╨* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_flatten_1_layer_call_and_return_conditional_losses_2009°
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0batch_normalization_1_3276batch_normalization_1_3278batch_normalization_1_3280batch_normalization_1_3282*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ╨*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1538є
reshape_3/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_reshape_3_layer_call_and_return_conditional_losses_2034ж
(gaussian_noise_1/StatefulPartitionedCallStatefulPartitionedCall"reshape_3/PartitionedCall:output:0'^gaussian_noise/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_gaussian_noise_1_layer_call_and_return_conditional_losses_2406Ї
activation_1/PartitionedCallPartitionedCall1gaussian_noise_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_activation_1_layer_call_and_return_conditional_losses_2047Ф
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0conv2d_1_3288conv2d_1_3290*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_2065▐
flatten_2/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         Z* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_flatten_2_layer_call_and_return_conditional_losses_2077ў
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0batch_normalization_2_3294batch_normalization_2_3296batch_normalization_2_3298batch_normalization_2_3300*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         Z*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1620є
reshape_4/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_reshape_4_layer_call_and_return_conditional_losses_2102и
(gaussian_noise_2/StatefulPartitionedCallStatefulPartitionedCall"reshape_4/PartitionedCall:output:0)^gaussian_noise_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_gaussian_noise_2_layer_call_and_return_conditional_losses_2356Ї
activation_2/PartitionedCallPartitionedCall1gaussian_noise_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_activation_2_layer_call_and_return_conditional_losses_2115┌
flatten_3/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         Z* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_flatten_3_layer_call_and_return_conditional_losses_2123ў
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0batch_normalization_3_3307batch_normalization_3_3309batch_normalization_3_3311batch_normalization_3_3313*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         Z*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1702С
dense/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0
dense_3316
dense_3318*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         %*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_2150┬
)dense/ActivityRegularizer/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *4
f/R-
+__inference_dense_activity_regularizer_1726u
dense/ActivityRegularizer/ShapeShape&dense/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:w
-dense/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/dense/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/dense/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╙
'dense/ActivityRegularizer/strided_sliceStridedSlice(dense/ActivityRegularizer/Shape:output:06dense/ActivityRegularizer/strided_slice/stack:output:08dense/ActivityRegularizer/strided_slice/stack_1:output:08dense/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskИ
dense/ActivityRegularizer/CastCast0dense/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: е
!dense/ActivityRegularizer/truedivRealDiv2dense/ActivityRegularizer/PartitionedCall:output:0"dense/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: с
activation_3/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         %* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_activation_3_layer_call_and_return_conditional_losses_2169Л
3CNNs_start/kernel/Regularizer/Square/ReadVariableOpReadVariableOpcnns_start_3251*&
_output_shapes
:		x*
dtype0Ь
$CNNs_start/kernel/Regularizer/SquareSquare;CNNs_start/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		x|
#CNNs_start/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             б
!CNNs_start/kernel/Regularizer/SumSum(CNNs_start/kernel/Regularizer/Square:y:0,CNNs_start/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#CNNs_start/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:г
!CNNs_start/kernel/Regularizer/mulMul,CNNs_start/kernel/Regularizer/mul/x:output:0*CNNs_start/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: Г
/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_3270*&
_output_shapes
:*
dtype0Ф
 conv2d/kernel/Regularizer/SquareSquare7conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:x
conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             Х
conv2d/kernel/Regularizer/SumSum$conv2d/kernel/Regularizer/Square:y:0(conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:Ч
conv2d/kernel/Regularizer/mulMul(conv2d/kernel/Regularizer/mul/x:output:0&conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: З
1conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_1_3288*&
_output_shapes
:*
dtype0Ш
"conv2d_1/kernel/Regularizer/SquareSquare9conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
!conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             Ы
conv2d_1/kernel/Regularizer/SumSum&conv2d_1/kernel/Regularizer/Square:y:0*conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:Э
conv2d_1/kernel/Regularizer/mulMul*conv2d_1/kernel/Regularizer/mul/x:output:0(conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: y
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp
dense_3316*
_output_shapes

:Z%*
dtype0К
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:Z%o
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Т
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:Ф
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: t
IdentityIdentity%activation_3/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         %e

Identity_1Identity%dense/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: ║
NoOpNoOp#^CNNs_start/StatefulPartitionedCall4^CNNs_start/kernel/Regularizer/Square/ReadVariableOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall^conv2d/StatefulPartitionedCall0^conv2d/kernel/Regularizer/Square/ReadVariableOp!^conv2d_1/StatefulPartitionedCall2^conv2d_1/kernel/Regularizer/Square/ReadVariableOp^dense/StatefulPartitionedCall/^dense/kernel/Regularizer/Square/ReadVariableOp'^gaussian_noise/StatefulPartitionedCall)^gaussian_noise_1/StatefulPartitionedCall)^gaussian_noise_2/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall1^photoreceptor_rods_reike/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*П
_input_shapes~
|:         ┤': : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
"CNNs_start/StatefulPartitionedCall"CNNs_start/StatefulPartitionedCall2j
3CNNs_start/kernel/Regularizer/Square/ReadVariableOp3CNNs_start/kernel/Regularizer/Square/ReadVariableOp2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2b
/conv2d/kernel/Regularizer/Square/ReadVariableOp/conv2d/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2f
1conv2d_1/kernel/Regularizer/Square/ReadVariableOp1conv2d_1/kernel/Regularizer/Square/ReadVariableOp2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2P
&gaussian_noise/StatefulPartitionedCall&gaussian_noise/StatefulPartitionedCall2T
(gaussian_noise_1/StatefulPartitionedCall(gaussian_noise_1/StatefulPartitionedCall2T
(gaussian_noise_2/StatefulPartitionedCall(gaussian_noise_2/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2d
0photoreceptor_rods_reike/StatefulPartitionedCall0photoreceptor_rods_reike/StatefulPartitionedCall:Y U
0
_output_shapes
:         ┤'
!
_user_specified_name	input_1
Ю
h
/__inference_gaussian_noise_1_layer_call_fn_4841

inputs
identityИвStatefulPartitionedCall╨
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_gaussian_noise_1_layer_call_and_return_conditional_losses_2406w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         	22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         	
 
_user_specified_nameinputs
■
п
B__inference_conv2d_1_layer_call_and_return_conditional_losses_2065

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpв1conv2d_1/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0▒
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
data_formatNCHW*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ф
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
data_formatNCHWШ
1conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ш
"conv2d_1/kernel/Regularizer/SquareSquare9conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
!conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             Ы
conv2d_1/kernel/Regularizer/SumSum&conv2d_1/kernel/Regularizer/Square:y:0*conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:Э
conv2d_1/kernel/Regularizer/mulMul*conv2d_1/kernel/Regularizer/mul/x:output:0(conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:         л
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp2^conv2d_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2f
1conv2d_1/kernel/Regularizer/Square/ReadVariableOp1conv2d_1/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:         	
 
_user_specified_nameinputs
ж	
i
J__inference_gaussian_noise_1_layer_call_and_return_conditional_losses_2406

inputs
identityИ;
ShapeShapeinputs*
T0*
_output_shapes
:W
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=С
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*/
_output_shapes
:         	*
dtype0Ю
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*/
_output_shapes
:         	Д
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*/
_output_shapes
:         	a
addAddV2inputsrandom_normal:z:0*
T0*/
_output_shapes
:         	W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:         	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         	:W S
/
_output_shapes
:         	
 
_user_specified_nameinputs
Яq
▄
__inference__traced_save_5420
file_prefix=
9savev2_photoreceptor_rods_reike_sigma_read_readvariableopF
Bsavev2_photoreceptor_rods_reike_sigma_scalefac_read_readvariableop;
7savev2_photoreceptor_rods_reike_phi_read_readvariableopD
@savev2_photoreceptor_rods_reike_phi_scalefac_read_readvariableop;
7savev2_photoreceptor_rods_reike_eta_read_readvariableopD
@savev2_photoreceptor_rods_reike_eta_scalefac_read_readvariableop<
8savev2_photoreceptor_rods_reike_beta_read_readvariableopE
Asavev2_photoreceptor_rods_reike_beta_scalefac_read_readvariableop@
<savev2_photoreceptor_rods_reike_cgmp2cur_read_readvariableop@
<savev2_photoreceptor_rods_reike_cgmphill_read_readvariableopI
Esavev2_photoreceptor_rods_reike_cgmphill_scalefac_read_readvariableop=
9savev2_photoreceptor_rods_reike_cdark_read_readvariableop@
<savev2_photoreceptor_rods_reike_betaslow_read_readvariableopI
Esavev2_photoreceptor_rods_reike_betaslow_scalefac_read_readvariableop@
<savev2_photoreceptor_rods_reike_hillcoef_read_readvariableopI
Esavev2_photoreceptor_rods_reike_hillcoef_scalefac_read_readvariableopD
@savev2_photoreceptor_rods_reike_hillaffinity_read_readvariableopM
Isavev2_photoreceptor_rods_reike_hillaffinity_scalefac_read_readvariableop=
9savev2_photoreceptor_rods_reike_gamma_read_readvariableopF
Bsavev2_photoreceptor_rods_reike_gamma_scalefac_read_readvariableop=
9savev2_photoreceptor_rods_reike_gdark_read_readvariableopF
Bsavev2_photoreceptor_rods_reike_gdark_scalefac_read_readvariableop8
4savev2_layer_normalization_gamma_read_readvariableop7
3savev2_layer_normalization_beta_read_readvariableop0
,savev2_cnns_start_kernel_read_readvariableop.
*savev2_cnns_start_bias_read_readvariableop8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop:
6savev2_batch_normalization_1_gamma_read_readvariableop9
5savev2_batch_normalization_1_beta_read_readvariableop@
<savev2_batch_normalization_1_moving_mean_read_readvariableopD
@savev2_batch_normalization_1_moving_variance_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop:
6savev2_batch_normalization_2_gamma_read_readvariableop9
5savev2_batch_normalization_2_beta_read_readvariableop@
<savev2_batch_normalization_2_moving_mean_read_readvariableopD
@savev2_batch_normalization_2_moving_variance_read_readvariableop:
6savev2_batch_normalization_3_gamma_read_readvariableop9
5savev2_batch_normalization_3_beta_read_readvariableop@
<savev2_batch_normalization_3_moving_mean_read_readvariableopD
@savev2_batch_normalization_3_moving_variance_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop&
"savev2_total_3_read_readvariableop&
"savev2_count_3_read_readvariableop
savev2_const

identity_1ИвMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/partБ
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ╕
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:9*
dtype0*с
value╫B╘9B5layer_with_weights-0/sigma/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-0/sigma_scaleFac/.ATTRIBUTES/VARIABLE_VALUEB3layer_with_weights-0/phi/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-0/phi_scaleFac/.ATTRIBUTES/VARIABLE_VALUEB3layer_with_weights-0/eta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-0/eta_scaleFac/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB=layer_with_weights-0/beta_scaleFac/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/cgmp2cur/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/cgmphill/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-0/cgmphill_scaleFac/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-0/cdark/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/betaSlow/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-0/betaSlow_scaleFac/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/hillcoef/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-0/hillcoef_scaleFac/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-0/hillaffinity/.ATTRIBUTES/VARIABLE_VALUEBElayer_with_weights-0/hillaffinity_scaleFac/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-0/gamma_scaleFac/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-0/gdark/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-0/gdark_scaleFac/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHр
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:9*
dtype0*Е
value|Bz9B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ·
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:09savev2_photoreceptor_rods_reike_sigma_read_readvariableopBsavev2_photoreceptor_rods_reike_sigma_scalefac_read_readvariableop7savev2_photoreceptor_rods_reike_phi_read_readvariableop@savev2_photoreceptor_rods_reike_phi_scalefac_read_readvariableop7savev2_photoreceptor_rods_reike_eta_read_readvariableop@savev2_photoreceptor_rods_reike_eta_scalefac_read_readvariableop8savev2_photoreceptor_rods_reike_beta_read_readvariableopAsavev2_photoreceptor_rods_reike_beta_scalefac_read_readvariableop<savev2_photoreceptor_rods_reike_cgmp2cur_read_readvariableop<savev2_photoreceptor_rods_reike_cgmphill_read_readvariableopEsavev2_photoreceptor_rods_reike_cgmphill_scalefac_read_readvariableop9savev2_photoreceptor_rods_reike_cdark_read_readvariableop<savev2_photoreceptor_rods_reike_betaslow_read_readvariableopEsavev2_photoreceptor_rods_reike_betaslow_scalefac_read_readvariableop<savev2_photoreceptor_rods_reike_hillcoef_read_readvariableopEsavev2_photoreceptor_rods_reike_hillcoef_scalefac_read_readvariableop@savev2_photoreceptor_rods_reike_hillaffinity_read_readvariableopIsavev2_photoreceptor_rods_reike_hillaffinity_scalefac_read_readvariableop9savev2_photoreceptor_rods_reike_gamma_read_readvariableopBsavev2_photoreceptor_rods_reike_gamma_scalefac_read_readvariableop9savev2_photoreceptor_rods_reike_gdark_read_readvariableopBsavev2_photoreceptor_rods_reike_gdark_scalefac_read_readvariableop4savev2_layer_normalization_gamma_read_readvariableop3savev2_layer_normalization_beta_read_readvariableop,savev2_cnns_start_kernel_read_readvariableop*savev2_cnns_start_bias_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop6savev2_batch_normalization_2_gamma_read_readvariableop5savev2_batch_normalization_2_beta_read_readvariableop<savev2_batch_normalization_2_moving_mean_read_readvariableop@savev2_batch_normalization_2_moving_variance_read_readvariableop6savev2_batch_normalization_3_gamma_read_readvariableop5savev2_batch_normalization_3_beta_read_readvariableop<savev2_batch_normalization_3_moving_mean_read_readvariableop@savev2_batch_normalization_3_moving_variance_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop"savev2_total_3_read_readvariableop"savev2_count_3_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *G
dtypes=
;29Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Л
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*с
_input_shapes╧
╠: :::::::::::::::::::::::x':x':		x::╨*:╨*:╨*:╨*:::╨:╨:╨:╨:::Z:Z:Z:Z:Z:Z:Z:Z:Z%:%: : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$	 

_output_shapes

::$
 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::($
"
_output_shapes
:x':($
"
_output_shapes
:x':,(
&
_output_shapes
:		x: 

_output_shapes
::!

_output_shapes	
:╨*:!

_output_shapes	
:╨*:!

_output_shapes	
:╨*:!

_output_shapes	
:╨*:,(
&
_output_shapes
::  

_output_shapes
::!!

_output_shapes	
:╨:!"

_output_shapes	
:╨:!#

_output_shapes	
:╨:!$

_output_shapes	
:╨:,%(
&
_output_shapes
:: &

_output_shapes
:: '

_output_shapes
:Z: (

_output_shapes
:Z: )

_output_shapes
:Z: *

_output_shapes
:Z: +

_output_shapes
:Z: ,

_output_shapes
:Z: -

_output_shapes
:Z: .

_output_shapes
:Z:$/ 

_output_shapes

:Z%: 0

_output_shapes
:%:1

_output_shapes
: :2

_output_shapes
: :3

_output_shapes
: :4

_output_shapes
: :5

_output_shapes
: :6

_output_shapes
: :7

_output_shapes
: :8

_output_shapes
: :9

_output_shapes
: 
╠
_
C__inference_reshape_3_layer_call_and_return_conditional_losses_2034

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :	й
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:         	`
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:         	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ╨:P L
(
_output_shapes
:         ╨
 
_user_specified_nameinputs
▒
│
__inference_loss_fn_1_5191R
8conv2d_kernel_regularizer_square_readvariableop_resource:
identityИв/conv2d/kernel/Regularizer/Square/ReadVariableOp░
/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp8conv2d_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:*
dtype0Ф
 conv2d/kernel/Regularizer/SquareSquare7conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:x
conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             Х
conv2d/kernel/Regularizer/SumSum$conv2d/kernel/Regularizer/Square:y:0(conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:Ч
conv2d/kernel/Regularizer/mulMul(conv2d/kernel/Regularizer/mul/x:output:0&conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentity!conv2d/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: x
NoOpNoOp0^conv2d/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2b
/conv2d/kernel/Regularizer/Square/ReadVariableOp/conv2d/kernel/Regularizer/Square/ReadVariableOp
Ў
╙
.__inference_PRFR_CNN2D_RODS_layer_call_fn_2297
input_1
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20: 

unknown_21:x' 

unknown_22:x'$

unknown_23:		x

unknown_24:

unknown_25:	╨*

unknown_26:	╨*

unknown_27:	╨*

unknown_28:	╨*$

unknown_29:

unknown_30:

unknown_31:	╨

unknown_32:	╨

unknown_33:	╨

unknown_34:	╨$

unknown_35:

unknown_36:

unknown_37:Z

unknown_38:Z

unknown_39:Z

unknown_40:Z

unknown_41:Z

unknown_42:Z

unknown_43:Z

unknown_44:Z

unknown_45:Z%

unknown_46:%
identityИвStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46*<
Tin5
321*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         %: *R
_read_only_resource_inputs4
20	
 !"#$%&'()*+,-./0*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_PRFR_CNN2D_RODS_layer_call_and_return_conditional_losses_2197o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         %`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*П
_input_shapes~
|:         ┤': : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:         ┤'
!
_user_specified_name	input_1
ж
б
?__inference_dense_layer_call_and_return_conditional_losses_5229

inputs0
matmul_readvariableop_resource:Z%-
biasadd_readvariableop_resource:%
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpв.dense/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:Z%*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         %r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:%*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         %Н
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:Z%*
dtype0К
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:Z%o
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Т
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:Ф
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         %и
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         Z: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:         Z
 
_user_specified_nameinputs
├
]
A__inference_flatten_layer_call_and_return_conditional_losses_4536

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"    P  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         ╨*Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         ╨*"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
ъ
b
F__inference_activation_2_layer_call_and_return_conditional_losses_2115

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:         b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
ъ
b
F__inference_activation_1_layer_call_and_return_conditional_losses_2047

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:         	b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:         	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         	:W S
/
_output_shapes
:         	
 
_user_specified_nameinputs
╠
о
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1573

inputs/
!batchnorm_readvariableop_resource:Z3
%batchnorm_mul_readvariableop_resource:Z1
#batchnorm_readvariableop_1_resource:Z1
#batchnorm_readvariableop_2_resource:Z
identityИвbatchnorm/ReadVariableOpвbatchnorm/ReadVariableOp_1вbatchnorm/ReadVariableOp_2вbatchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:Z*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:ZP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:Z~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:Z*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Zc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:         Zz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:Z*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:Zz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:Z*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Zr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:         Zb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:         Z║
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         Z: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:         Z
 
_user_specified_nameinputs
▐
▓
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1491

inputs0
!batchnorm_readvariableop_resource:	╨4
%batchnorm_mul_readvariableop_resource:	╨2
#batchnorm_readvariableop_1_resource:	╨2
#batchnorm_readvariableop_2_resource:	╨
identityИвbatchnorm/ReadVariableOpвbatchnorm/ReadVariableOp_1вbatchnorm/ReadVariableOp_2вbatchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:╨*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:╨Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:╨
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:╨*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:╨d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         ╨{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:╨*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:╨{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:╨*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:╨s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         ╨c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:         ╨║
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ╨: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:         ╨
 
_user_specified_nameinputs
╞
H
,__inference_max_pooling2d_layer_call_fn_4645

inputs
identity╜
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_1966h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
ж
c
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_4650

inputs
identity╣
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
data_formatNCHW*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
д
G
+__inference_activation_3_layer_call_fn_5164

inputs
identity┤
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         %* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_activation_3_layer_call_and_return_conditional_losses_2169`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         %"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         %:O K
'
_output_shapes
:         %
 
_user_specified_nameinputs
Йq
А
__inference_riekeModel_1187	
x_fun	
sigma
phi
eta
cgmp2cur:
cgmphill
cdark:
beta
betaslow
hillcoef
hillaffinity	
gamma	
gdark
identityИвReadVariableOpвReadVariableOp_1вReadVariableOp_2вReadVariableOp_3вmul/ReadVariableOpвmul_2/ReadVariableOpвtruediv_1/ReadVariableOpвwhileD
powPowgdarkcgmphill*
T0*
_output_shapes

:[
mul/ReadVariableOpReadVariableOpcgmp2cur*
_output_shapes

:*
dtype0X
mulMulpow:z:0mul/ReadVariableOp:value:0*
T0*
_output_shapes

:N
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @X
truedivRealDivmul:z:0truediv/y:output:0*
T0*
_output_shapes

:L
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @T
mul_1Mulmul_1/x:output:0truediv:z:0*
T0*
_output_shapes

:a
truediv_1/ReadVariableOpReadVariableOpcgmp2cur*
_output_shapes

:*
dtype0j
	truediv_1RealDiv	mul_1:z:0 truediv_1/ReadVariableOp:value:0*
T0*
_output_shapes

:P
truediv_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?]
	truediv_2RealDivtruediv_2/x:output:0cgmphill*
T0*
_output_shapes

:S
pow_1Powtruediv_1:z:0truediv_2:z:0*
T0*
_output_shapes

:Z
mul_2/ReadVariableOpReadVariableOpcdark*
_output_shapes

:*
dtype0Y
mul_2Mulbetamul_2/ReadVariableOp:value:0*
T0*
_output_shapes

:U
	truediv_3RealDiv	mul_2:z:0truediv:z:0*
T0*
_output_shapes

:G
	truediv_4RealDivetaphi*
T0*
_output_shapes

:O
mul_3Multruediv_4:z:0	pow_1:z:0*
T0*
_output_shapes

:T
ReadVariableOpReadVariableOpcdark*
_output_shapes

:*
dtype0c
	truediv_5RealDivReadVariableOp:value:0hillaffinity*
T0*
_output_shapes

:N
pow_2Powtruediv_5:z:0hillcoef*
T0*
_output_shapes

:J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?P
addAddV2add/x:output:0	pow_2:z:0*
T0*
_output_shapes

:I
mul_4Mul	mul_3:z:0add:z:0*
T0*
_output_shapes

:P
range/startConst*
_output_shapes
: *
dtype0*
valueB
 *    P
range/limitConst*
_output_shapes
: *
dtype0*
valueB
 *  4CP
range/deltaConst*
_output_shapes
: *
dtype0*
valueB
 *  А?y
rangeRangerange/start:output:0range/limit:output:0range/delta:output:0*

Tidx0*
_output_shapes	
:┤L
mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *o<T
mul_5Mulrange:output:0mul_5/y:output:0*
T0*
_output_shapes	
:┤h
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            j
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           j
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         №
strided_sliceStridedSlicex_funstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         Т	*

begin_mask*
end_mask*
shrink_axis_maskL
mul_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *    i
mul_6Mulstrided_slice:output:0mul_6/y:output:0*
T0*(
_output_shapes
:         Т	W
add_1AddV2	pow_1:z:0	mul_6:z:0*
T0*(
_output_shapes
:         Т	E
mul_7Mul	pow_1:z:0eta*
T0*
_output_shapes

:M
	truediv_6RealDiv	mul_7:z:0phi*
T0*
_output_shapes

:j
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            l
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           l
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Д
strided_slice_1StridedSlicex_funstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         Т	*

begin_mask*
end_mask*
shrink_axis_maskL
mul_8/yConst*
_output_shapes
: *
dtype0*
valueB
 *    k
mul_8Mulstrided_slice_1:output:0mul_8/y:output:0*
T0*(
_output_shapes
:         Т	[
add_2AddV2truediv_6:z:0	mul_8:z:0*
T0*(
_output_shapes
:         Т	j
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"            l
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           l
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Д
strided_slice_2StridedSlicex_funstrided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         Т	*

begin_mask*
end_mask*
shrink_axis_maskL
mul_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *    k
mul_9Mulstrided_slice_2:output:0mul_9/y:output:0*
T0*(
_output_shapes
:         Т	V
ReadVariableOp_1ReadVariableOpcdark*
_output_shapes

:*
dtype0f
add_3AddV2ReadVariableOp_1:value:0	mul_9:z:0*
T0*(
_output_shapes
:         Т	j
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            l
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           l
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Д
strided_slice_3StridedSlicex_funstrided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         Т	*

begin_mask*
end_mask*
shrink_axis_maskM
mul_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *    m
mul_10Mulstrided_slice_3:output:0mul_10/y:output:0*
T0*(
_output_shapes
:         Т	V
ReadVariableOp_2ReadVariableOpcdark*
_output_shapes

:*
dtype0g
add_4AddV2ReadVariableOp_2:value:0
mul_10:z:0*
T0*(
_output_shapes
:         Т	j
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"            l
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           l
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Д
strided_slice_4StridedSlicex_funstrided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         Т	*

begin_mask*
end_mask*
shrink_axis_maska
mul_11Mulstrided_slice_4:output:0gamma*
T0*(
_output_shapes
:         Т	Z
	truediv_7RealDiv
mul_11:z:0sigma*
T0*(
_output_shapes
:         Т	U
add_5AddV2etatruediv_7:z:0*
T0*(
_output_shapes
:         Т	W
	truediv_8RealDiv	add_5:z:0phi*
T0*(
_output_shapes
:         Т	f
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ]
TensorArrayV2/num_elementsConst*
_output_shapes
: *
dtype0*
value
B :┤┐
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0#TensorArrayV2/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥j
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*!
valueB"            l
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           l
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Д
strided_slice_5StridedSlicex_funstrided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         Т	*

begin_mask*
end_mask*
shrink_axis_maskM
mul_12/yConst*
_output_shapes
: *
dtype0*
valueB
 *    m
mul_12Mulstrided_slice_5:output:0mul_12/y:output:0*
T0*(
_output_shapes
:         Т	l
*TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ╥
$TensorArrayV2Write/TensorListSetItemTensorListSetItemTensorArrayV2:handle:03TensorArrayV2Write/TensorListSetItem/index:output:0
mul_12:z:0*
_output_shapes
: *
element_dtype0:щш╥O
range_1/startConst*
_output_shapes
: *
dtype0*
value	B :P
range_1/limitConst*
_output_shapes
: *
dtype0*
value
B :┤O
range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :u
range_1Rangerange_1/start:output:0range_1/limit:output:0range_1/delta:output:0*
_output_shapes	
:│[
subSubrange_1/limit:output:0range_1/start:output:0*
T0*
_output_shapes
: V
floordivFloorDivsub:z:0range_1/delta:output:0*
T0*
_output_shapes
: Q
modFloorModsub:z:0range_1/delta:output:0*
T0*
_output_shapes
: L

zeros_likeConst*
_output_shapes
: *
dtype0*
value	B : S
NotEqualNotEqualmod:z:0zeros_like:output:0*
T0*
_output_shapes
: J
CastCastNotEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
: G
add_6AddV2floordiv:z:0Cast:y:0*
T0*
_output_shapes
: N
zeros_like_1Const*
_output_shapes
: *
dtype0*
value	B : U
MaximumMaximum	add_6:z:0zeros_like_1:output:0*
T0*
_output_shapes
: c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ├
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0range_1/start:output:0TensorArrayV2:handle:0	add_3:z:0	add_1:z:0truediv_8:z:0truediv_7:z:0	add_2:z:0range_1/limit:output:0sigmax_fungammaetaphicgmphillcgmp2curtruediv_3:z:0betahillaffinityhillcoef	mul_4:z:0range_1/delta:output:0* 
T
2*
_lower_using_switch_merge(*
_num_original_outputs*Е
_output_shapesЄ
я: : : : :         Т	:         Т	:         Т	:         Т	:         Т	: ::         ┤Т	::::: :::::: *#
_read_only_resource_inputs
*
_stateful_parallelism( *
bodyR
while_body_1040*
condR
while_cond_1039*Д
output_shapesЄ
я: : : : :         Т	:         Т	:         Т	:         Т	:         Т	: ::         ┤Т	::::: :::::: Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Т  ┘
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:┤         Т	*
element_dtype0*
num_elements┤c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ф
	transpose	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose/perm:output:0*
T0*-
_output_shapes
:         ┤Т	]
pow_3Powtranspose:y:0cgmphill*
T0*-
_output_shapes
:         ┤Т	Y
ReadVariableOp_3ReadVariableOpcgmp2cur*
_output_shapes

:*
dtype0j
mul_13MulReadVariableOp_3:value:0	pow_3:z:0*
T0*-
_output_shapes
:         ┤Т	N
NegNeg
mul_13:z:0*
T0*-
_output_shapes
:         ┤Т	P
truediv_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @k
	truediv_9RealDivNeg:y:0truediv_9/y:output:0*
T0*-
_output_shapes
:         ┤Т	b
IdentityIdentitytruediv_9:z:0^NoOp*
T0*-
_output_shapes
:         ┤Т	▀
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^mul/ReadVariableOp^mul_2/ReadVariableOp^truediv_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ц
_input_shapesД
Б:         ┤Т	:::: :: ::::::2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32(
mul/ReadVariableOpmul/ReadVariableOp2,
mul_2/ReadVariableOpmul_2/ReadVariableOp24
truediv_1/ReadVariableOptruediv_1/ReadVariableOp2
whilewhile:T P
-
_output_shapes
:         ┤Т	

_user_specified_nameX_fun:EA

_output_shapes

:

_user_specified_namesigma:C?

_output_shapes

:

_user_specified_namephi:C?

_output_shapes

:

_user_specified_nameeta:($
"
_user_specified_name
cgmp2cur:HD

_output_shapes

:
"
_user_specified_name
cgmphill:%!

_user_specified_namecdark:D@

_output_shapes

:

_user_specified_namebeta:HD

_output_shapes

:
"
_user_specified_name
betaSlow:H	D

_output_shapes

:
"
_user_specified_name
hillcoef:L
H

_output_shapes

:
&
_user_specified_namehillaffinity:EA

_output_shapes

:

_user_specified_namegamma:EA

_output_shapes

:

_user_specified_namegdark
ис
С&
 __inference__traced_restore_5598
file_prefixA
/assignvariableop_photoreceptor_rods_reike_sigma:L
:assignvariableop_1_photoreceptor_rods_reike_sigma_scalefac:A
/assignvariableop_2_photoreceptor_rods_reike_phi:J
8assignvariableop_3_photoreceptor_rods_reike_phi_scalefac:A
/assignvariableop_4_photoreceptor_rods_reike_eta:J
8assignvariableop_5_photoreceptor_rods_reike_eta_scalefac:B
0assignvariableop_6_photoreceptor_rods_reike_beta:K
9assignvariableop_7_photoreceptor_rods_reike_beta_scalefac:F
4assignvariableop_8_photoreceptor_rods_reike_cgmp2cur:F
4assignvariableop_9_photoreceptor_rods_reike_cgmphill:P
>assignvariableop_10_photoreceptor_rods_reike_cgmphill_scalefac:D
2assignvariableop_11_photoreceptor_rods_reike_cdark:G
5assignvariableop_12_photoreceptor_rods_reike_betaslow:P
>assignvariableop_13_photoreceptor_rods_reike_betaslow_scalefac:G
5assignvariableop_14_photoreceptor_rods_reike_hillcoef:P
>assignvariableop_15_photoreceptor_rods_reike_hillcoef_scalefac:K
9assignvariableop_16_photoreceptor_rods_reike_hillaffinity:T
Bassignvariableop_17_photoreceptor_rods_reike_hillaffinity_scalefac:D
2assignvariableop_18_photoreceptor_rods_reike_gamma:M
;assignvariableop_19_photoreceptor_rods_reike_gamma_scalefac:D
2assignvariableop_20_photoreceptor_rods_reike_gdark:M
;assignvariableop_21_photoreceptor_rods_reike_gdark_scalefac:C
-assignvariableop_22_layer_normalization_gamma:x'B
,assignvariableop_23_layer_normalization_beta:x'?
%assignvariableop_24_cnns_start_kernel:		x1
#assignvariableop_25_cnns_start_bias:<
-assignvariableop_26_batch_normalization_gamma:	╨*;
,assignvariableop_27_batch_normalization_beta:	╨*B
3assignvariableop_28_batch_normalization_moving_mean:	╨*F
7assignvariableop_29_batch_normalization_moving_variance:	╨*;
!assignvariableop_30_conv2d_kernel:-
assignvariableop_31_conv2d_bias:>
/assignvariableop_32_batch_normalization_1_gamma:	╨=
.assignvariableop_33_batch_normalization_1_beta:	╨D
5assignvariableop_34_batch_normalization_1_moving_mean:	╨H
9assignvariableop_35_batch_normalization_1_moving_variance:	╨=
#assignvariableop_36_conv2d_1_kernel:/
!assignvariableop_37_conv2d_1_bias:=
/assignvariableop_38_batch_normalization_2_gamma:Z<
.assignvariableop_39_batch_normalization_2_beta:ZC
5assignvariableop_40_batch_normalization_2_moving_mean:ZG
9assignvariableop_41_batch_normalization_2_moving_variance:Z=
/assignvariableop_42_batch_normalization_3_gamma:Z<
.assignvariableop_43_batch_normalization_3_beta:ZC
5assignvariableop_44_batch_normalization_3_moving_mean:ZG
9assignvariableop_45_batch_normalization_3_moving_variance:Z2
 assignvariableop_46_dense_kernel:Z%,
assignvariableop_47_dense_bias:%#
assignvariableop_48_total: #
assignvariableop_49_count: %
assignvariableop_50_total_1: %
assignvariableop_51_count_1: %
assignvariableop_52_total_2: %
assignvariableop_53_count_2: %
assignvariableop_54_total_3: %
assignvariableop_55_count_3: 
identity_57ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_27вAssignVariableOp_28вAssignVariableOp_29вAssignVariableOp_3вAssignVariableOp_30вAssignVariableOp_31вAssignVariableOp_32вAssignVariableOp_33вAssignVariableOp_34вAssignVariableOp_35вAssignVariableOp_36вAssignVariableOp_37вAssignVariableOp_38вAssignVariableOp_39вAssignVariableOp_4вAssignVariableOp_40вAssignVariableOp_41вAssignVariableOp_42вAssignVariableOp_43вAssignVariableOp_44вAssignVariableOp_45вAssignVariableOp_46вAssignVariableOp_47вAssignVariableOp_48вAssignVariableOp_49вAssignVariableOp_5вAssignVariableOp_50вAssignVariableOp_51вAssignVariableOp_52вAssignVariableOp_53вAssignVariableOp_54вAssignVariableOp_55вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9╗
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:9*
dtype0*с
value╫B╘9B5layer_with_weights-0/sigma/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-0/sigma_scaleFac/.ATTRIBUTES/VARIABLE_VALUEB3layer_with_weights-0/phi/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-0/phi_scaleFac/.ATTRIBUTES/VARIABLE_VALUEB3layer_with_weights-0/eta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-0/eta_scaleFac/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB=layer_with_weights-0/beta_scaleFac/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/cgmp2cur/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/cgmphill/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-0/cgmphill_scaleFac/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-0/cdark/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/betaSlow/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-0/betaSlow_scaleFac/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/hillcoef/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-0/hillcoef_scaleFac/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-0/hillaffinity/.ATTRIBUTES/VARIABLE_VALUEBElayer_with_weights-0/hillaffinity_scaleFac/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-0/gamma_scaleFac/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-0/gdark/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-0/gdark_scaleFac/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHу
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:9*
dtype0*Е
value|Bz9B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ╛
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*·
_output_shapesч
ф:::::::::::::::::::::::::::::::::::::::::::::::::::::::::*G
dtypes=
;29[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOpAssignVariableOp/assignvariableop_photoreceptor_rods_reike_sigmaIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:й
AssignVariableOp_1AssignVariableOp:assignvariableop_1_photoreceptor_rods_reike_sigma_scalefacIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_2AssignVariableOp/assignvariableop_2_photoreceptor_rods_reike_phiIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:з
AssignVariableOp_3AssignVariableOp8assignvariableop_3_photoreceptor_rods_reike_phi_scalefacIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_4AssignVariableOp/assignvariableop_4_photoreceptor_rods_reike_etaIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:з
AssignVariableOp_5AssignVariableOp8assignvariableop_5_photoreceptor_rods_reike_eta_scalefacIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_6AssignVariableOp0assignvariableop_6_photoreceptor_rods_reike_betaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:и
AssignVariableOp_7AssignVariableOp9assignvariableop_7_photoreceptor_rods_reike_beta_scalefacIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:г
AssignVariableOp_8AssignVariableOp4assignvariableop_8_photoreceptor_rods_reike_cgmp2curIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:г
AssignVariableOp_9AssignVariableOp4assignvariableop_9_photoreceptor_rods_reike_cgmphillIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:п
AssignVariableOp_10AssignVariableOp>assignvariableop_10_photoreceptor_rods_reike_cgmphill_scalefacIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:г
AssignVariableOp_11AssignVariableOp2assignvariableop_11_photoreceptor_rods_reike_cdarkIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:ж
AssignVariableOp_12AssignVariableOp5assignvariableop_12_photoreceptor_rods_reike_betaslowIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:п
AssignVariableOp_13AssignVariableOp>assignvariableop_13_photoreceptor_rods_reike_betaslow_scalefacIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:ж
AssignVariableOp_14AssignVariableOp5assignvariableop_14_photoreceptor_rods_reike_hillcoefIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:п
AssignVariableOp_15AssignVariableOp>assignvariableop_15_photoreceptor_rods_reike_hillcoef_scalefacIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:к
AssignVariableOp_16AssignVariableOp9assignvariableop_16_photoreceptor_rods_reike_hillaffinityIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:│
AssignVariableOp_17AssignVariableOpBassignvariableop_17_photoreceptor_rods_reike_hillaffinity_scalefacIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:г
AssignVariableOp_18AssignVariableOp2assignvariableop_18_photoreceptor_rods_reike_gammaIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:м
AssignVariableOp_19AssignVariableOp;assignvariableop_19_photoreceptor_rods_reike_gamma_scalefacIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:г
AssignVariableOp_20AssignVariableOp2assignvariableop_20_photoreceptor_rods_reike_gdarkIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:м
AssignVariableOp_21AssignVariableOp;assignvariableop_21_photoreceptor_rods_reike_gdark_scalefacIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_22AssignVariableOp-assignvariableop_22_layer_normalization_gammaIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_23AssignVariableOp,assignvariableop_23_layer_normalization_betaIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_24AssignVariableOp%assignvariableop_24_cnns_start_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_25AssignVariableOp#assignvariableop_25_cnns_start_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_26AssignVariableOp-assignvariableop_26_batch_normalization_gammaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_27AssignVariableOp,assignvariableop_27_batch_normalization_betaIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:д
AssignVariableOp_28AssignVariableOp3assignvariableop_28_batch_normalization_moving_meanIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:и
AssignVariableOp_29AssignVariableOp7assignvariableop_29_batch_normalization_moving_varianceIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_30AssignVariableOp!assignvariableop_30_conv2d_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_31AssignVariableOpassignvariableop_31_conv2d_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:а
AssignVariableOp_32AssignVariableOp/assignvariableop_32_batch_normalization_1_gammaIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_33AssignVariableOp.assignvariableop_33_batch_normalization_1_betaIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:ж
AssignVariableOp_34AssignVariableOp5assignvariableop_34_batch_normalization_1_moving_meanIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:к
AssignVariableOp_35AssignVariableOp9assignvariableop_35_batch_normalization_1_moving_varianceIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_36AssignVariableOp#assignvariableop_36_conv2d_1_kernelIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_37AssignVariableOp!assignvariableop_37_conv2d_1_biasIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:а
AssignVariableOp_38AssignVariableOp/assignvariableop_38_batch_normalization_2_gammaIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_39AssignVariableOp.assignvariableop_39_batch_normalization_2_betaIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:ж
AssignVariableOp_40AssignVariableOp5assignvariableop_40_batch_normalization_2_moving_meanIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:к
AssignVariableOp_41AssignVariableOp9assignvariableop_41_batch_normalization_2_moving_varianceIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:а
AssignVariableOp_42AssignVariableOp/assignvariableop_42_batch_normalization_3_gammaIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_43AssignVariableOp.assignvariableop_43_batch_normalization_3_betaIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:ж
AssignVariableOp_44AssignVariableOp5assignvariableop_44_batch_normalization_3_moving_meanIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:к
AssignVariableOp_45AssignVariableOp9assignvariableop_45_batch_normalization_3_moving_varianceIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_46AssignVariableOp assignvariableop_46_dense_kernelIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_47AssignVariableOpassignvariableop_47_dense_biasIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_48AssignVariableOpassignvariableop_48_totalIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_49AssignVariableOpassignvariableop_49_countIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_50AssignVariableOpassignvariableop_50_total_1Identity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_51AssignVariableOpassignvariableop_51_count_1Identity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_52AssignVariableOpassignvariableop_52_total_2Identity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_53AssignVariableOpassignvariableop_53_count_2Identity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_54AssignVariableOpassignvariableop_54_total_3Identity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_55AssignVariableOpassignvariableop_55_count_3Identity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 Я

Identity_56Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_57IdentityIdentity_56:output:0^NoOp_1*
T0*
_output_shapes
: М

NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_57Identity_57:output:0*Е
_input_shapest
r: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
 

й
__inference_loss_fn_3_5213I
7dense_kernel_regularizer_square_readvariableop_resource:Z%
identityИв.dense/kernel/Regularizer/Square/ReadVariableOpж
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp7dense_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:Z%*
dtype0К
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:Z%o
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Т
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:Ф
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ^
IdentityIdentity dense/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: w
NoOpNoOp/^dense/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp
э
Ю
)__inference_CNNs_start_layer_call_fn_4509

inputs!
unknown:		x
	unknown_0:
identityИвStatefulPartitionedCallф
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_CNNs_start_layer_call_and_return_conditional_losses_1923w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         x': : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         x'
 
_user_specified_nameinputs
ж
c
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_1464

inputs
identity╣
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
data_formatNCHW*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
▐
л
@__inference_conv2d_layer_call_and_return_conditional_losses_4721

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpв/conv2d/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0▒
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         	*
data_formatNCHW*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ф
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         	*
data_formatNCHWЦ
/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ф
 conv2d/kernel/Regularizer/SquareSquare7conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:x
conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             Х
conv2d/kernel/Regularizer/SumSum$conv2d/kernel/Regularizer/Square:y:0(conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:Ч
conv2d/kernel/Regularizer/mulMul(conv2d/kernel/Regularizer/mul/x:output:0&conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:         	й
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp0^conv2d/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2b
/conv2d/kernel/Regularizer/Square/ReadVariableOp/conv2d/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
╝
D
(__inference_reshape_1_layer_call_fn_4445

inputs
identity║
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ┤'* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_reshape_1_layer_call_and_return_conditional_losses_1869i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:         ┤'"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:         ┤Т	:U Q
-
_output_shapes
:         ┤Т	
 
_user_specified_nameinputs
╚
I
-__inference_gaussian_noise_layer_call_fn_4660

inputs
identity╛
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_gaussian_noise_layer_call_and_return_conditional_losses_1972h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
╓
b
F__inference_activation_3_layer_call_and_return_conditional_losses_2169

inputs
identityN
SoftplusSoftplusinputs*
T0*'
_output_shapes
:         %^
IdentityIdentitySoftplus:activations:0*
T0*'
_output_shapes
:         %"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         %:O K
'
_output_shapes
:         %
 
_user_specified_nameinputs
╩
_
C__inference_reshape_4_layer_call_and_return_conditional_losses_2102

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :й
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:         `
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         Z:O K
'
_output_shapes
:         Z
 
_user_specified_nameinputs
╕
B
&__inference_reshape_layer_call_fn_4317

inputs
identity╡
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         ┤Т	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_reshape_layer_call_and_return_conditional_losses_1746f
IdentityIdentityPartitionedCall:output:0*
T0*-
_output_shapes
:         ┤Т	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ┤':X T
0
_output_shapes
:         ┤'
 
_user_specified_nameinputs
Р%
ш
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1620

inputs5
'assignmovingavg_readvariableop_resource:Z7
)assignmovingavg_1_readvariableop_resource:Z3
%batchnorm_mul_readvariableop_resource:Z/
!batchnorm_readvariableop_resource:Z
identityИвAssignMovingAvgвAssignMovingAvg/ReadVariableOpвAssignMovingAvg_1в AssignMovingAvg_1/ReadVariableOpвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:Z*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:ZЗ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:         Zl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ю
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:Z*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:Z*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:Z*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<В
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:Z*
dtype0Б
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:Zx
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:Zм
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<Ж
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:Z*
dtype0З
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:Z~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:Z┤
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:ZP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:Z~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:Z*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Zc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:         Zh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:Zv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:Z*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Zr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:         Zb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:         Zъ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         Z: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:         Z
 
_user_specified_nameinputs
╗
С
$__inference_dense_layer_call_fn_5148

inputs
unknown:Z%
	unknown_0:%
identityИвStatefulPartitionedCall╫
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         %*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_2150o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         %`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         Z: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         Z
 
_user_specified_nameinputs
╝Є
о0
I__inference_PRFR_CNN2D_RODS_layer_call_and_return_conditional_losses_4312

inputsB
0photoreceptor_rods_reike_readvariableop_resource:F
4photoreceptor_rods_reike_mul_readvariableop_resource:D
2photoreceptor_rods_reike_readvariableop_1_resource:H
6photoreceptor_rods_reike_mul_1_readvariableop_resource:D
2photoreceptor_rods_reike_readvariableop_2_resource:H
6photoreceptor_rods_reike_mul_2_readvariableop_resource:D
2photoreceptor_rods_reike_readvariableop_3_resource:H
6photoreceptor_rods_reike_mul_3_readvariableop_resource:D
2photoreceptor_rods_reike_readvariableop_4_resource:H
6photoreceptor_rods_reike_mul_4_readvariableop_resource:D
2photoreceptor_rods_reike_readvariableop_5_resource:H
6photoreceptor_rods_reike_mul_5_readvariableop_resource:D
2photoreceptor_rods_reike_readvariableop_6_resource:H
6photoreceptor_rods_reike_mul_6_readvariableop_resource:D
2photoreceptor_rods_reike_readvariableop_7_resource:H
6photoreceptor_rods_reike_mul_7_readvariableop_resource:D
2photoreceptor_rods_reike_readvariableop_8_resource:H
6photoreceptor_rods_reike_mul_8_readvariableop_resource:D
2photoreceptor_rods_reike_readvariableop_9_resource:H
6photoreceptor_rods_reike_mul_9_readvariableop_resource:/
photoreceptor_rods_reike_4025:/
photoreceptor_rods_reike_4027:I
3layer_normalization_reshape_readvariableop_resource:x'K
5layer_normalization_reshape_1_readvariableop_resource:x'C
)cnns_start_conv2d_readvariableop_resource:		x8
*cnns_start_biasadd_readvariableop_resource:J
;batch_normalization_assignmovingavg_readvariableop_resource:	╨*L
=batch_normalization_assignmovingavg_1_readvariableop_resource:	╨*H
9batch_normalization_batchnorm_mul_readvariableop_resource:	╨*D
5batch_normalization_batchnorm_readvariableop_resource:	╨*?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:L
=batch_normalization_1_assignmovingavg_readvariableop_resource:	╨N
?batch_normalization_1_assignmovingavg_1_readvariableop_resource:	╨J
;batch_normalization_1_batchnorm_mul_readvariableop_resource:	╨F
7batch_normalization_1_batchnorm_readvariableop_resource:	╨A
'conv2d_1_conv2d_readvariableop_resource:6
(conv2d_1_biasadd_readvariableop_resource:K
=batch_normalization_2_assignmovingavg_readvariableop_resource:ZM
?batch_normalization_2_assignmovingavg_1_readvariableop_resource:ZI
;batch_normalization_2_batchnorm_mul_readvariableop_resource:ZE
7batch_normalization_2_batchnorm_readvariableop_resource:ZK
=batch_normalization_3_assignmovingavg_readvariableop_resource:ZM
?batch_normalization_3_assignmovingavg_1_readvariableop_resource:ZI
;batch_normalization_3_batchnorm_mul_readvariableop_resource:ZE
7batch_normalization_3_batchnorm_readvariableop_resource:Z6
$dense_matmul_readvariableop_resource:Z%3
%dense_biasadd_readvariableop_resource:%
identity

identity_1Ив!CNNs_start/BiasAdd/ReadVariableOpв CNNs_start/Conv2D/ReadVariableOpв3CNNs_start/kernel/Regularizer/Square/ReadVariableOpв#batch_normalization/AssignMovingAvgв2batch_normalization/AssignMovingAvg/ReadVariableOpв%batch_normalization/AssignMovingAvg_1в4batch_normalization/AssignMovingAvg_1/ReadVariableOpв,batch_normalization/batchnorm/ReadVariableOpв0batch_normalization/batchnorm/mul/ReadVariableOpв%batch_normalization_1/AssignMovingAvgв4batch_normalization_1/AssignMovingAvg/ReadVariableOpв'batch_normalization_1/AssignMovingAvg_1в6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpв.batch_normalization_1/batchnorm/ReadVariableOpв2batch_normalization_1/batchnorm/mul/ReadVariableOpв%batch_normalization_2/AssignMovingAvgв4batch_normalization_2/AssignMovingAvg/ReadVariableOpв'batch_normalization_2/AssignMovingAvg_1в6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpв.batch_normalization_2/batchnorm/ReadVariableOpв2batch_normalization_2/batchnorm/mul/ReadVariableOpв%batch_normalization_3/AssignMovingAvgв4batch_normalization_3/AssignMovingAvg/ReadVariableOpв'batch_normalization_3/AssignMovingAvg_1в6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpв.batch_normalization_3/batchnorm/ReadVariableOpв2batch_normalization_3/batchnorm/mul/ReadVariableOpвconv2d/BiasAdd/ReadVariableOpвconv2d/Conv2D/ReadVariableOpв/conv2d/kernel/Regularizer/Square/ReadVariableOpвconv2d_1/BiasAdd/ReadVariableOpвconv2d_1/Conv2D/ReadVariableOpв1conv2d_1/kernel/Regularizer/Square/ReadVariableOpвdense/BiasAdd/ReadVariableOpвdense/MatMul/ReadVariableOpв.dense/kernel/Regularizer/Square/ReadVariableOpв*layer_normalization/Reshape/ReadVariableOpв,layer_normalization/Reshape_1/ReadVariableOpв'photoreceptor_rods_reike/ReadVariableOpв)photoreceptor_rods_reike/ReadVariableOp_1в)photoreceptor_rods_reike/ReadVariableOp_2в)photoreceptor_rods_reike/ReadVariableOp_3в)photoreceptor_rods_reike/ReadVariableOp_4в)photoreceptor_rods_reike/ReadVariableOp_5в)photoreceptor_rods_reike/ReadVariableOp_6в)photoreceptor_rods_reike/ReadVariableOp_7в)photoreceptor_rods_reike/ReadVariableOp_8в)photoreceptor_rods_reike/ReadVariableOp_9в0photoreceptor_rods_reike/StatefulPartitionedCallв+photoreceptor_rods_reike/mul/ReadVariableOpв-photoreceptor_rods_reike/mul_1/ReadVariableOpв-photoreceptor_rods_reike/mul_2/ReadVariableOpв-photoreceptor_rods_reike/mul_3/ReadVariableOpв-photoreceptor_rods_reike/mul_4/ReadVariableOpв-photoreceptor_rods_reike/mul_5/ReadVariableOpв-photoreceptor_rods_reike/mul_6/ReadVariableOpв-photoreceptor_rods_reike/mul_7/ReadVariableOpв-photoreceptor_rods_reike/mul_8/ReadVariableOpв-photoreceptor_rods_reike/mul_9/ReadVariableOpC
reshape/ShapeShapeinputs*
T0*
_output_shapes
:e
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:∙
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :┤Z
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :Т	п
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:z
reshape/ReshapeReshapeinputsreshape/Reshape/shape:output:0*
T0*-
_output_shapes
:         ┤Т	Ш
'photoreceptor_rods_reike/ReadVariableOpReadVariableOp0photoreceptor_rods_reike_readvariableop_resource*
_output_shapes

:*
dtype0а
+photoreceptor_rods_reike/mul/ReadVariableOpReadVariableOp4photoreceptor_rods_reike_mul_readvariableop_resource*
_output_shapes

:*
dtype0▓
photoreceptor_rods_reike/mulMul/photoreceptor_rods_reike/ReadVariableOp:value:03photoreceptor_rods_reike/mul/ReadVariableOp:value:0*
T0*
_output_shapes

:Ь
)photoreceptor_rods_reike/ReadVariableOp_1ReadVariableOp2photoreceptor_rods_reike_readvariableop_1_resource*
_output_shapes

:*
dtype0д
-photoreceptor_rods_reike/mul_1/ReadVariableOpReadVariableOp6photoreceptor_rods_reike_mul_1_readvariableop_resource*
_output_shapes

:*
dtype0╕
photoreceptor_rods_reike/mul_1Mul1photoreceptor_rods_reike/ReadVariableOp_1:value:05photoreceptor_rods_reike/mul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:Ь
)photoreceptor_rods_reike/ReadVariableOp_2ReadVariableOp2photoreceptor_rods_reike_readvariableop_2_resource*
_output_shapes

:*
dtype0д
-photoreceptor_rods_reike/mul_2/ReadVariableOpReadVariableOp6photoreceptor_rods_reike_mul_2_readvariableop_resource*
_output_shapes

:*
dtype0╕
photoreceptor_rods_reike/mul_2Mul1photoreceptor_rods_reike/ReadVariableOp_2:value:05photoreceptor_rods_reike/mul_2/ReadVariableOp:value:0*
T0*
_output_shapes

:Ь
)photoreceptor_rods_reike/ReadVariableOp_3ReadVariableOp2photoreceptor_rods_reike_readvariableop_3_resource*
_output_shapes

:*
dtype0д
-photoreceptor_rods_reike/mul_3/ReadVariableOpReadVariableOp6photoreceptor_rods_reike_mul_3_readvariableop_resource*
_output_shapes

:*
dtype0╕
photoreceptor_rods_reike/mul_3Mul1photoreceptor_rods_reike/ReadVariableOp_3:value:05photoreceptor_rods_reike/mul_3/ReadVariableOp:value:0*
T0*
_output_shapes

:Ь
)photoreceptor_rods_reike/ReadVariableOp_4ReadVariableOp2photoreceptor_rods_reike_readvariableop_4_resource*
_output_shapes

:*
dtype0д
-photoreceptor_rods_reike/mul_4/ReadVariableOpReadVariableOp6photoreceptor_rods_reike_mul_4_readvariableop_resource*
_output_shapes

:*
dtype0╕
photoreceptor_rods_reike/mul_4Mul1photoreceptor_rods_reike/ReadVariableOp_4:value:05photoreceptor_rods_reike/mul_4/ReadVariableOp:value:0*
T0*
_output_shapes

:Ь
)photoreceptor_rods_reike/ReadVariableOp_5ReadVariableOp2photoreceptor_rods_reike_readvariableop_5_resource*
_output_shapes

:*
dtype0д
-photoreceptor_rods_reike/mul_5/ReadVariableOpReadVariableOp6photoreceptor_rods_reike_mul_5_readvariableop_resource*
_output_shapes

:*
dtype0╕
photoreceptor_rods_reike/mul_5Mul1photoreceptor_rods_reike/ReadVariableOp_5:value:05photoreceptor_rods_reike/mul_5/ReadVariableOp:value:0*
T0*
_output_shapes

:Ь
)photoreceptor_rods_reike/ReadVariableOp_6ReadVariableOp2photoreceptor_rods_reike_readvariableop_6_resource*
_output_shapes

:*
dtype0д
-photoreceptor_rods_reike/mul_6/ReadVariableOpReadVariableOp6photoreceptor_rods_reike_mul_6_readvariableop_resource*
_output_shapes

:*
dtype0╕
photoreceptor_rods_reike/mul_6Mul1photoreceptor_rods_reike/ReadVariableOp_6:value:05photoreceptor_rods_reike/mul_6/ReadVariableOp:value:0*
T0*
_output_shapes

:Ь
)photoreceptor_rods_reike/ReadVariableOp_7ReadVariableOp2photoreceptor_rods_reike_readvariableop_7_resource*
_output_shapes

:*
dtype0д
-photoreceptor_rods_reike/mul_7/ReadVariableOpReadVariableOp6photoreceptor_rods_reike_mul_7_readvariableop_resource*
_output_shapes

:*
dtype0╕
photoreceptor_rods_reike/mul_7Mul1photoreceptor_rods_reike/ReadVariableOp_7:value:05photoreceptor_rods_reike/mul_7/ReadVariableOp:value:0*
T0*
_output_shapes

:Ь
)photoreceptor_rods_reike/ReadVariableOp_8ReadVariableOp2photoreceptor_rods_reike_readvariableop_8_resource*
_output_shapes

:*
dtype0д
-photoreceptor_rods_reike/mul_8/ReadVariableOpReadVariableOp6photoreceptor_rods_reike_mul_8_readvariableop_resource*
_output_shapes

:*
dtype0╕
photoreceptor_rods_reike/mul_8Mul1photoreceptor_rods_reike/ReadVariableOp_8:value:05photoreceptor_rods_reike/mul_8/ReadVariableOp:value:0*
T0*
_output_shapes

:g
"photoreceptor_rods_reike/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Aе
 photoreceptor_rods_reike/truedivRealDiv"photoreceptor_rods_reike/mul_8:z:0+photoreceptor_rods_reike/truediv/y:output:0*
T0*
_output_shapes

:Ь
)photoreceptor_rods_reike/ReadVariableOp_9ReadVariableOp2photoreceptor_rods_reike_readvariableop_9_resource*
_output_shapes

:*
dtype0д
-photoreceptor_rods_reike/mul_9/ReadVariableOpReadVariableOp6photoreceptor_rods_reike_mul_9_readvariableop_resource*
_output_shapes

:*
dtype0╕
photoreceptor_rods_reike/mul_9Mul1photoreceptor_rods_reike/ReadVariableOp_9:value:05photoreceptor_rods_reike/mul_9/ReadVariableOp:value:0*
T0*
_output_shapes

:А
0photoreceptor_rods_reike/StatefulPartitionedCallStatefulPartitionedCallreshape/Reshape:output:0 photoreceptor_rods_reike/mul:z:0"photoreceptor_rods_reike/mul_1:z:0"photoreceptor_rods_reike/mul_2:z:0photoreceptor_rods_reike_4025"photoreceptor_rods_reike/mul_3:z:0photoreceptor_rods_reike_4027"photoreceptor_rods_reike/mul_4:z:0"photoreceptor_rods_reike/mul_5:z:0"photoreceptor_rods_reike/mul_6:z:0"photoreceptor_rods_reike/mul_7:z:0$photoreceptor_rods_reike/truediv:z:0"photoreceptor_rods_reike/mul_9:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         ┤Т	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *$
fR
__inference_riekeModel_1187x
reshape_1/ShapeShape9photoreceptor_rods_reike/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:g
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Г
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :┤[
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :'█
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0"reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:┤
reshape_1/ReshapeReshape9photoreceptor_rods_reike/StatefulPartitionedCall:output:0 reshape_1/Reshape/shape:output:0*
T0*0
_output_shapes
:         ┤'Е
,tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"    <           З
.tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                З
.tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            ф
&tf.__operators__.getitem/strided_sliceStridedSlicereshape_1/Reshape:output:05tf.__operators__.getitem/strided_slice/stack:output:07tf.__operators__.getitem/strided_slice/stack_1:output:07tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         x'*

begin_mask*
end_maskЗ
2layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         с
 layer_normalization/moments/meanMean/tf.__operators__.getitem/strided_slice:output:0;layer_normalization/moments/mean/reduction_indices:output:0*
T0*/
_output_shapes
:         *
	keep_dims(Э
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*/
_output_shapes
:         р
-layer_normalization/moments/SquaredDifferenceSquaredDifference/tf.__operators__.getitem/strided_slice:output:01layer_normalization/moments/StopGradient:output:0*
T0*/
_output_shapes
:         x'Л
6layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         ы
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
T0*/
_output_shapes
:         *
	keep_dims(в
*layer_normalization/Reshape/ReadVariableOpReadVariableOp3layer_normalization_reshape_readvariableop_resource*"
_output_shapes
:x'*
dtype0z
!layer_normalization/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   x      '   ╖
layer_normalization/ReshapeReshape2layer_normalization/Reshape/ReadVariableOp:value:0*layer_normalization/Reshape/shape:output:0*
T0*&
_output_shapes
:x'ж
,layer_normalization/Reshape_1/ReadVariableOpReadVariableOp5layer_normalization_reshape_1_readvariableop_resource*"
_output_shapes
:x'*
dtype0|
#layer_normalization/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   x      '   ╜
layer_normalization/Reshape_1Reshape4layer_normalization/Reshape_1/ReadVariableOp:value:0,layer_normalization/Reshape_1/shape:output:0*
T0*&
_output_shapes
:x'h
#layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Х┐╓3┴
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*/
_output_shapes
:         Н
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*/
_output_shapes
:         ▒
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:0$layer_normalization/Reshape:output:0*
T0*/
_output_shapes
:         x'╝
#layer_normalization/batchnorm/mul_1Mul/tf.__operators__.getitem/strided_slice:output:0%layer_normalization/batchnorm/mul:z:0*
T0*/
_output_shapes
:         x'╢
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*/
_output_shapes
:         x'│
!layer_normalization/batchnorm/subSub&layer_normalization/Reshape_1:output:0'layer_normalization/batchnorm/mul_2:z:0*
T0*/
_output_shapes
:         x'╢
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*/
_output_shapes
:         x'Т
 CNNs_start/Conv2D/ReadVariableOpReadVariableOp)cnns_start_conv2d_readvariableop_resource*&
_output_shapes
:		x*
dtype0ш
CNNs_start/Conv2DConv2D'layer_normalization/batchnorm/add_1:z:0(CNNs_start/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
data_formatNCHW*
paddingVALID*
strides
И
!CNNs_start/BiasAdd/ReadVariableOpReadVariableOp*cnns_start_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0╡
CNNs_start/BiasAddBiasAddCNNs_start/Conv2D:output:0)CNNs_start/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
data_formatNCHW^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"    P  В
flatten/ReshapeReshapeCNNs_start/BiasAdd:output:0flatten/Const:output:0*
T0*(
_output_shapes
:         ╨*|
2batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ║
 batch_normalization/moments/meanMeanflatten/Reshape:output:0;batch_normalization/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	╨**
	keep_dims(Н
(batch_normalization/moments/StopGradientStopGradient)batch_normalization/moments/mean:output:0*
T0*
_output_shapes
:	╨*┬
-batch_normalization/moments/SquaredDifferenceSquaredDifferenceflatten/Reshape:output:01batch_normalization/moments/StopGradient:output:0*
T0*(
_output_shapes
:         ╨*А
6batch_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: █
$batch_normalization/moments/varianceMean1batch_normalization/moments/SquaredDifference:z:0?batch_normalization/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	╨**
	keep_dims(Ц
#batch_normalization/moments/SqueezeSqueeze)batch_normalization/moments/mean:output:0*
T0*
_output_shapes	
:╨**
squeeze_dims
 Ь
%batch_normalization/moments/Squeeze_1Squeeze-batch_normalization/moments/variance:output:0*
T0*
_output_shapes	
:╨**
squeeze_dims
 n
)batch_normalization/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<л
2batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOp;batch_normalization_assignmovingavg_readvariableop_resource*
_output_shapes	
:╨**
dtype0╛
'batch_normalization/AssignMovingAvg/subSub:batch_normalization/AssignMovingAvg/ReadVariableOp:value:0,batch_normalization/moments/Squeeze:output:0*
T0*
_output_shapes	
:╨*╡
'batch_normalization/AssignMovingAvg/mulMul+batch_normalization/AssignMovingAvg/sub:z:02batch_normalization/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:╨*№
#batch_normalization/AssignMovingAvgAssignSubVariableOp;batch_normalization_assignmovingavg_readvariableop_resource+batch_normalization/AssignMovingAvg/mul:z:03^batch_normalization/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0p
+batch_normalization/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<п
4batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOp=batch_normalization_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:╨**
dtype0─
)batch_normalization/AssignMovingAvg_1/subSub<batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:0.batch_normalization/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:╨*╗
)batch_normalization/AssignMovingAvg_1/mulMul-batch_normalization/AssignMovingAvg_1/sub:z:04batch_normalization/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:╨*Д
%batch_normalization/AssignMovingAvg_1AssignSubVariableOp=batch_normalization_assignmovingavg_1_readvariableop_resource-batch_normalization/AssignMovingAvg_1/mul:z:05^batch_normalization/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0h
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:о
!batch_normalization/batchnorm/addAddV2.batch_normalization/moments/Squeeze_1:output:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:╨*y
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:╨*з
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:╨**
dtype0▒
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:╨*Ю
#batch_normalization/batchnorm/mul_1Mulflatten/Reshape:output:0%batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:         ╨*е
#batch_normalization/batchnorm/mul_2Mul,batch_normalization/moments/Squeeze:output:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:╨*Я
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:╨**
dtype0н
!batch_normalization/batchnorm/subSub4batch_normalization/batchnorm/ReadVariableOp:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:╨*п
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:         ╨*f
reshape_2/ShapeShape'batch_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:g
reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Г
reshape_2/strided_sliceStridedSlicereshape_2/Shape:output:0&reshape_2/strided_slice/stack:output:0(reshape_2/strided_slice/stack_1:output:0(reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :█
reshape_2/Reshape/shapePack reshape_2/strided_slice:output:0"reshape_2/Reshape/shape/1:output:0"reshape_2/Reshape/shape/2:output:0"reshape_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:б
reshape_2/ReshapeReshape'batch_normalization/batchnorm/add_1:z:0 reshape_2/Reshape/shape:output:0*
T0*/
_output_shapes
:         └
max_pooling2d/MaxPoolMaxPoolreshape_2/Reshape:output:0*/
_output_shapes
:         *
data_formatNCHW*
ksize
*
paddingVALID*
strides
b
gaussian_noise/ShapeShapemax_pooling2d/MaxPool:output:0*
T0*
_output_shapes
:f
!gaussian_noise/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    h
#gaussian_noise/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=п
1gaussian_noise/random_normal/RandomStandardNormalRandomStandardNormalgaussian_noise/Shape:output:0*
T0*/
_output_shapes
:         *
dtype0╦
 gaussian_noise/random_normal/mulMul:gaussian_noise/random_normal/RandomStandardNormal:output:0,gaussian_noise/random_normal/stddev:output:0*
T0*/
_output_shapes
:         ▒
gaussian_noise/random_normalAddV2$gaussian_noise/random_normal/mul:z:0*gaussian_noise/random_normal/mean:output:0*
T0*/
_output_shapes
:         Ч
gaussian_noise/addAddV2max_pooling2d/MaxPool:output:0 gaussian_noise/random_normal:z:0*
T0*/
_output_shapes
:         i
activation/ReluRelugaussian_noise/add:z:0*
T0*/
_output_shapes
:         К
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0╓
conv2d/Conv2DConv2Dactivation/Relu:activations:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         	*
data_formatNCHW*
paddingVALID*
strides
А
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0й
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         	*
data_formatNCHW`
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"    ╨  В
flatten_1/ReshapeReshapeconv2d/BiasAdd:output:0flatten_1/Const:output:0*
T0*(
_output_shapes
:         ╨~
4batch_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: └
"batch_normalization_1/moments/meanMeanflatten_1/Reshape:output:0=batch_normalization_1/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	╨*
	keep_dims(С
*batch_normalization_1/moments/StopGradientStopGradient+batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes
:	╨╚
/batch_normalization_1/moments/SquaredDifferenceSquaredDifferenceflatten_1/Reshape:output:03batch_normalization_1/moments/StopGradient:output:0*
T0*(
_output_shapes
:         ╨В
8batch_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: с
&batch_normalization_1/moments/varianceMean3batch_normalization_1/moments/SquaredDifference:z:0Abatch_normalization_1/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	╨*
	keep_dims(Ъ
%batch_normalization_1/moments/SqueezeSqueeze+batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes	
:╨*
squeeze_dims
 а
'batch_normalization_1/moments/Squeeze_1Squeeze/batch_normalization_1/moments/variance:output:0*
T0*
_output_shapes	
:╨*
squeeze_dims
 p
+batch_normalization_1/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<п
4batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_1_assignmovingavg_readvariableop_resource*
_output_shapes	
:╨*
dtype0─
)batch_normalization_1/AssignMovingAvg/subSub<batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_1/moments/Squeeze:output:0*
T0*
_output_shapes	
:╨╗
)batch_normalization_1/AssignMovingAvg/mulMul-batch_normalization_1/AssignMovingAvg/sub:z:04batch_normalization_1/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:╨Д
%batch_normalization_1/AssignMovingAvgAssignSubVariableOp=batch_normalization_1_assignmovingavg_readvariableop_resource-batch_normalization_1/AssignMovingAvg/mul:z:05^batch_normalization_1/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0r
-batch_normalization_1/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<│
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_1_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:╨*
dtype0╩
+batch_normalization_1/AssignMovingAvg_1/subSub>batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_1/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:╨┴
+batch_normalization_1/AssignMovingAvg_1/mulMul/batch_normalization_1/AssignMovingAvg_1/sub:z:06batch_normalization_1/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:╨М
'batch_normalization_1/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_1_assignmovingavg_1_readvariableop_resource/batch_normalization_1/AssignMovingAvg_1/mul:z:07^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0j
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:┤
#batch_normalization_1/batchnorm/addAddV20batch_normalization_1/moments/Squeeze_1:output:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:╨}
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:╨л
2batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:╨*
dtype0╖
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:0:batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:╨д
%batch_normalization_1/batchnorm/mul_1Mulflatten_1/Reshape:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:         ╨л
%batch_normalization_1/batchnorm/mul_2Mul.batch_normalization_1/moments/Squeeze:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:╨г
.batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:╨*
dtype0│
#batch_normalization_1/batchnorm/subSub6batch_normalization_1/batchnorm/ReadVariableOp:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:╨╡
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:         ╨h
reshape_3/ShapeShape)batch_normalization_1/batchnorm/add_1:z:0*
T0*
_output_shapes
:g
reshape_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Г
reshape_3/strided_sliceStridedSlicereshape_3/Shape:output:0&reshape_3/strided_slice/stack:output:0(reshape_3/strided_slice/stack_1:output:0(reshape_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
reshape_3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_3/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_3/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :	█
reshape_3/Reshape/shapePack reshape_3/strided_slice:output:0"reshape_3/Reshape/shape/1:output:0"reshape_3/Reshape/shape/2:output:0"reshape_3/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:г
reshape_3/ReshapeReshape)batch_normalization_1/batchnorm/add_1:z:0 reshape_3/Reshape/shape:output:0*
T0*/
_output_shapes
:         	`
gaussian_noise_1/ShapeShapereshape_3/Reshape:output:0*
T0*
_output_shapes
:h
#gaussian_noise_1/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    j
%gaussian_noise_1/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=│
3gaussian_noise_1/random_normal/RandomStandardNormalRandomStandardNormalgaussian_noise_1/Shape:output:0*
T0*/
_output_shapes
:         	*
dtype0╤
"gaussian_noise_1/random_normal/mulMul<gaussian_noise_1/random_normal/RandomStandardNormal:output:0.gaussian_noise_1/random_normal/stddev:output:0*
T0*/
_output_shapes
:         	╖
gaussian_noise_1/random_normalAddV2&gaussian_noise_1/random_normal/mul:z:0,gaussian_noise_1/random_normal/mean:output:0*
T0*/
_output_shapes
:         	Ч
gaussian_noise_1/addAddV2reshape_3/Reshape:output:0"gaussian_noise_1/random_normal:z:0*
T0*/
_output_shapes
:         	m
activation_1/ReluRelugaussian_noise_1/add:z:0*
T0*/
_output_shapes
:         	О
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0▄
conv2d_1/Conv2DConv2Dactivation_1/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
data_formatNCHW*
paddingVALID*
strides
Д
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0п
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
data_formatNCHW`
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"    Z   Г
flatten_2/ReshapeReshapeconv2d_1/BiasAdd:output:0flatten_2/Const:output:0*
T0*'
_output_shapes
:         Z~
4batch_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ┐
"batch_normalization_2/moments/meanMeanflatten_2/Reshape:output:0=batch_normalization_2/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:Z*
	keep_dims(Р
*batch_normalization_2/moments/StopGradientStopGradient+batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes

:Z╟
/batch_normalization_2/moments/SquaredDifferenceSquaredDifferenceflatten_2/Reshape:output:03batch_normalization_2/moments/StopGradient:output:0*
T0*'
_output_shapes
:         ZВ
8batch_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: р
&batch_normalization_2/moments/varianceMean3batch_normalization_2/moments/SquaredDifference:z:0Abatch_normalization_2/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:Z*
	keep_dims(Щ
%batch_normalization_2/moments/SqueezeSqueeze+batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes
:Z*
squeeze_dims
 Я
'batch_normalization_2/moments/Squeeze_1Squeeze/batch_normalization_2/moments/variance:output:0*
T0*
_output_shapes
:Z*
squeeze_dims
 p
+batch_normalization_2/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<о
4batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_2_assignmovingavg_readvariableop_resource*
_output_shapes
:Z*
dtype0├
)batch_normalization_2/AssignMovingAvg/subSub<batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_2/moments/Squeeze:output:0*
T0*
_output_shapes
:Z║
)batch_normalization_2/AssignMovingAvg/mulMul-batch_normalization_2/AssignMovingAvg/sub:z:04batch_normalization_2/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:ZД
%batch_normalization_2/AssignMovingAvgAssignSubVariableOp=batch_normalization_2_assignmovingavg_readvariableop_resource-batch_normalization_2/AssignMovingAvg/mul:z:05^batch_normalization_2/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0r
-batch_normalization_2/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<▓
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_2_assignmovingavg_1_readvariableop_resource*
_output_shapes
:Z*
dtype0╔
+batch_normalization_2/AssignMovingAvg_1/subSub>batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_2/moments/Squeeze_1:output:0*
T0*
_output_shapes
:Z└
+batch_normalization_2/AssignMovingAvg_1/mulMul/batch_normalization_2/AssignMovingAvg_1/sub:z:06batch_normalization_2/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:ZМ
'batch_normalization_2/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_2_assignmovingavg_1_readvariableop_resource/batch_normalization_2/AssignMovingAvg_1/mul:z:07^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0j
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:│
#batch_normalization_2/batchnorm/addAddV20batch_normalization_2/moments/Squeeze_1:output:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:Z|
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:Zк
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:Z*
dtype0╢
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Zг
%batch_normalization_2/batchnorm/mul_1Mulflatten_2/Reshape:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:         Zк
%batch_normalization_2/batchnorm/mul_2Mul.batch_normalization_2/moments/Squeeze:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:Zв
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:Z*
dtype0▓
#batch_normalization_2/batchnorm/subSub6batch_normalization_2/batchnorm/ReadVariableOp:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Z┤
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:         Zh
reshape_4/ShapeShape)batch_normalization_2/batchnorm/add_1:z:0*
T0*
_output_shapes
:g
reshape_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Г
reshape_4/strided_sliceStridedSlicereshape_4/Shape:output:0&reshape_4/strided_slice/stack:output:0(reshape_4/strided_slice/stack_1:output:0(reshape_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
reshape_4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_4/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_4/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :█
reshape_4/Reshape/shapePack reshape_4/strided_slice:output:0"reshape_4/Reshape/shape/1:output:0"reshape_4/Reshape/shape/2:output:0"reshape_4/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:г
reshape_4/ReshapeReshape)batch_normalization_2/batchnorm/add_1:z:0 reshape_4/Reshape/shape:output:0*
T0*/
_output_shapes
:         `
gaussian_noise_2/ShapeShapereshape_4/Reshape:output:0*
T0*
_output_shapes
:h
#gaussian_noise_2/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    j
%gaussian_noise_2/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=│
3gaussian_noise_2/random_normal/RandomStandardNormalRandomStandardNormalgaussian_noise_2/Shape:output:0*
T0*/
_output_shapes
:         *
dtype0╤
"gaussian_noise_2/random_normal/mulMul<gaussian_noise_2/random_normal/RandomStandardNormal:output:0.gaussian_noise_2/random_normal/stddev:output:0*
T0*/
_output_shapes
:         ╖
gaussian_noise_2/random_normalAddV2&gaussian_noise_2/random_normal/mul:z:0,gaussian_noise_2/random_normal/mean:output:0*
T0*/
_output_shapes
:         Ч
gaussian_noise_2/addAddV2reshape_4/Reshape:output:0"gaussian_noise_2/random_normal:z:0*
T0*/
_output_shapes
:         m
activation_2/ReluRelugaussian_noise_2/add:z:0*
T0*/
_output_shapes
:         `
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"    Z   Й
flatten_3/ReshapeReshapeactivation_2/Relu:activations:0flatten_3/Const:output:0*
T0*'
_output_shapes
:         Z~
4batch_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ┐
"batch_normalization_3/moments/meanMeanflatten_3/Reshape:output:0=batch_normalization_3/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:Z*
	keep_dims(Р
*batch_normalization_3/moments/StopGradientStopGradient+batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes

:Z╟
/batch_normalization_3/moments/SquaredDifferenceSquaredDifferenceflatten_3/Reshape:output:03batch_normalization_3/moments/StopGradient:output:0*
T0*'
_output_shapes
:         ZВ
8batch_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: р
&batch_normalization_3/moments/varianceMean3batch_normalization_3/moments/SquaredDifference:z:0Abatch_normalization_3/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:Z*
	keep_dims(Щ
%batch_normalization_3/moments/SqueezeSqueeze+batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes
:Z*
squeeze_dims
 Я
'batch_normalization_3/moments/Squeeze_1Squeeze/batch_normalization_3/moments/variance:output:0*
T0*
_output_shapes
:Z*
squeeze_dims
 p
+batch_normalization_3/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<о
4batch_normalization_3/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_3_assignmovingavg_readvariableop_resource*
_output_shapes
:Z*
dtype0├
)batch_normalization_3/AssignMovingAvg/subSub<batch_normalization_3/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_3/moments/Squeeze:output:0*
T0*
_output_shapes
:Z║
)batch_normalization_3/AssignMovingAvg/mulMul-batch_normalization_3/AssignMovingAvg/sub:z:04batch_normalization_3/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:ZД
%batch_normalization_3/AssignMovingAvgAssignSubVariableOp=batch_normalization_3_assignmovingavg_readvariableop_resource-batch_normalization_3/AssignMovingAvg/mul:z:05^batch_normalization_3/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0r
-batch_normalization_3/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<▓
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_3_assignmovingavg_1_readvariableop_resource*
_output_shapes
:Z*
dtype0╔
+batch_normalization_3/AssignMovingAvg_1/subSub>batch_normalization_3/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_3/moments/Squeeze_1:output:0*
T0*
_output_shapes
:Z└
+batch_normalization_3/AssignMovingAvg_1/mulMul/batch_normalization_3/AssignMovingAvg_1/sub:z:06batch_normalization_3/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:ZМ
'batch_normalization_3/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_3_assignmovingavg_1_readvariableop_resource/batch_normalization_3/AssignMovingAvg_1/mul:z:07^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0j
%batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:│
#batch_normalization_3/batchnorm/addAddV20batch_normalization_3/moments/Squeeze_1:output:0.batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
:Z|
%batch_normalization_3/batchnorm/RsqrtRsqrt'batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:Zк
2batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:Z*
dtype0╢
#batch_normalization_3/batchnorm/mulMul)batch_normalization_3/batchnorm/Rsqrt:y:0:batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Zг
%batch_normalization_3/batchnorm/mul_1Mulflatten_3/Reshape:output:0'batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:         Zк
%batch_normalization_3/batchnorm/mul_2Mul.batch_normalization_3/moments/Squeeze:output:0'batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:Zв
.batch_normalization_3/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:Z*
dtype0▓
#batch_normalization_3/batchnorm/subSub6batch_normalization_3/batchnorm/ReadVariableOp:value:0)batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Z┤
%batch_normalization_3/batchnorm/add_1AddV2)batch_normalization_3/batchnorm/mul_1:z:0'batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:         ZА
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:Z%*
dtype0Ш
dense/MatMulMatMul)batch_normalization_3/batchnorm/add_1:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         %~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:%*
dtype0И
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         %n
dense/ActivityRegularizer/AbsAbsdense/BiasAdd:output:0*
T0*'
_output_shapes
:         %p
dense/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Т
dense/ActivityRegularizer/SumSum!dense/ActivityRegularizer/Abs:y:0(dense/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: d
dense/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:Ч
dense/ActivityRegularizer/mulMul(dense/ActivityRegularizer/mul/x:output:0&dense/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: e
dense/ActivityRegularizer/ShapeShapedense/BiasAdd:output:0*
T0*
_output_shapes
:w
-dense/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/dense/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/dense/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╙
'dense/ActivityRegularizer/strided_sliceStridedSlice(dense/ActivityRegularizer/Shape:output:06dense/ActivityRegularizer/strided_slice/stack:output:08dense/ActivityRegularizer/strided_slice/stack_1:output:08dense/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskИ
dense/ActivityRegularizer/CastCast0dense/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: Ф
!dense/ActivityRegularizer/truedivRealDiv!dense/ActivityRegularizer/mul:z:0"dense/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: k
activation_3/SoftplusSoftplusdense/BiasAdd:output:0*
T0*'
_output_shapes
:         %е
3CNNs_start/kernel/Regularizer/Square/ReadVariableOpReadVariableOp)cnns_start_conv2d_readvariableop_resource*&
_output_shapes
:		x*
dtype0Ь
$CNNs_start/kernel/Regularizer/SquareSquare;CNNs_start/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		x|
#CNNs_start/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             б
!CNNs_start/kernel/Regularizer/SumSum(CNNs_start/kernel/Regularizer/Square:y:0,CNNs_start/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#CNNs_start/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:г
!CNNs_start/kernel/Regularizer/mulMul,CNNs_start/kernel/Regularizer/mul/x:output:0*CNNs_start/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: Э
/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ф
 conv2d/kernel/Regularizer/SquareSquare7conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:x
conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             Х
conv2d/kernel/Regularizer/SumSum$conv2d/kernel/Regularizer/Square:y:0(conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:Ч
conv2d/kernel/Regularizer/mulMul(conv2d/kernel/Regularizer/mul/x:output:0&conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: б
1conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ш
"conv2d_1/kernel/Regularizer/SquareSquare9conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
!conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             Ы
conv2d_1/kernel/Regularizer/SumSum&conv2d_1/kernel/Regularizer/Square:y:0*conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:Э
conv2d_1/kernel/Regularizer/mulMul*conv2d_1/kernel/Regularizer/mul/x:output:0(conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: У
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:Z%*
dtype0К
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:Z%o
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Т
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:Ф
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: r
IdentityIdentity#activation_3/Softplus:activations:0^NoOp*
T0*'
_output_shapes
:         %e

Identity_1Identity%dense/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: ╨
NoOpNoOp"^CNNs_start/BiasAdd/ReadVariableOp!^CNNs_start/Conv2D/ReadVariableOp4^CNNs_start/kernel/Regularizer/Square/ReadVariableOp$^batch_normalization/AssignMovingAvg3^batch_normalization/AssignMovingAvg/ReadVariableOp&^batch_normalization/AssignMovingAvg_15^batch_normalization/AssignMovingAvg_1/ReadVariableOp-^batch_normalization/batchnorm/ReadVariableOp1^batch_normalization/batchnorm/mul/ReadVariableOp&^batch_normalization_1/AssignMovingAvg5^batch_normalization_1/AssignMovingAvg/ReadVariableOp(^batch_normalization_1/AssignMovingAvg_17^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp3^batch_normalization_1/batchnorm/mul/ReadVariableOp&^batch_normalization_2/AssignMovingAvg5^batch_normalization_2/AssignMovingAvg/ReadVariableOp(^batch_normalization_2/AssignMovingAvg_17^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_2/batchnorm/ReadVariableOp3^batch_normalization_2/batchnorm/mul/ReadVariableOp&^batch_normalization_3/AssignMovingAvg5^batch_normalization_3/AssignMovingAvg/ReadVariableOp(^batch_normalization_3/AssignMovingAvg_17^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_3/batchnorm/ReadVariableOp3^batch_normalization_3/batchnorm/mul/ReadVariableOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp0^conv2d/kernel/Regularizer/Square/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp2^conv2d_1/kernel/Regularizer/Square/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp+^layer_normalization/Reshape/ReadVariableOp-^layer_normalization/Reshape_1/ReadVariableOp(^photoreceptor_rods_reike/ReadVariableOp*^photoreceptor_rods_reike/ReadVariableOp_1*^photoreceptor_rods_reike/ReadVariableOp_2*^photoreceptor_rods_reike/ReadVariableOp_3*^photoreceptor_rods_reike/ReadVariableOp_4*^photoreceptor_rods_reike/ReadVariableOp_5*^photoreceptor_rods_reike/ReadVariableOp_6*^photoreceptor_rods_reike/ReadVariableOp_7*^photoreceptor_rods_reike/ReadVariableOp_8*^photoreceptor_rods_reike/ReadVariableOp_91^photoreceptor_rods_reike/StatefulPartitionedCall,^photoreceptor_rods_reike/mul/ReadVariableOp.^photoreceptor_rods_reike/mul_1/ReadVariableOp.^photoreceptor_rods_reike/mul_2/ReadVariableOp.^photoreceptor_rods_reike/mul_3/ReadVariableOp.^photoreceptor_rods_reike/mul_4/ReadVariableOp.^photoreceptor_rods_reike/mul_5/ReadVariableOp.^photoreceptor_rods_reike/mul_6/ReadVariableOp.^photoreceptor_rods_reike/mul_7/ReadVariableOp.^photoreceptor_rods_reike/mul_8/ReadVariableOp.^photoreceptor_rods_reike/mul_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*П
_input_shapes~
|:         ┤': : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2F
!CNNs_start/BiasAdd/ReadVariableOp!CNNs_start/BiasAdd/ReadVariableOp2D
 CNNs_start/Conv2D/ReadVariableOp CNNs_start/Conv2D/ReadVariableOp2j
3CNNs_start/kernel/Regularizer/Square/ReadVariableOp3CNNs_start/kernel/Regularizer/Square/ReadVariableOp2J
#batch_normalization/AssignMovingAvg#batch_normalization/AssignMovingAvg2h
2batch_normalization/AssignMovingAvg/ReadVariableOp2batch_normalization/AssignMovingAvg/ReadVariableOp2N
%batch_normalization/AssignMovingAvg_1%batch_normalization/AssignMovingAvg_12l
4batch_normalization/AssignMovingAvg_1/ReadVariableOp4batch_normalization/AssignMovingAvg_1/ReadVariableOp2\
,batch_normalization/batchnorm/ReadVariableOp,batch_normalization/batchnorm/ReadVariableOp2d
0batch_normalization/batchnorm/mul/ReadVariableOp0batch_normalization/batchnorm/mul/ReadVariableOp2N
%batch_normalization_1/AssignMovingAvg%batch_normalization_1/AssignMovingAvg2l
4batch_normalization_1/AssignMovingAvg/ReadVariableOp4batch_normalization_1/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_1/AssignMovingAvg_1'batch_normalization_1/AssignMovingAvg_12p
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_1/batchnorm/ReadVariableOp.batch_normalization_1/batchnorm/ReadVariableOp2h
2batch_normalization_1/batchnorm/mul/ReadVariableOp2batch_normalization_1/batchnorm/mul/ReadVariableOp2N
%batch_normalization_2/AssignMovingAvg%batch_normalization_2/AssignMovingAvg2l
4batch_normalization_2/AssignMovingAvg/ReadVariableOp4batch_normalization_2/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_2/AssignMovingAvg_1'batch_normalization_2/AssignMovingAvg_12p
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_2/batchnorm/ReadVariableOp.batch_normalization_2/batchnorm/ReadVariableOp2h
2batch_normalization_2/batchnorm/mul/ReadVariableOp2batch_normalization_2/batchnorm/mul/ReadVariableOp2N
%batch_normalization_3/AssignMovingAvg%batch_normalization_3/AssignMovingAvg2l
4batch_normalization_3/AssignMovingAvg/ReadVariableOp4batch_normalization_3/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_3/AssignMovingAvg_1'batch_normalization_3/AssignMovingAvg_12p
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_3/batchnorm/ReadVariableOp.batch_normalization_3/batchnorm/ReadVariableOp2h
2batch_normalization_3/batchnorm/mul/ReadVariableOp2batch_normalization_3/batchnorm/mul/ReadVariableOp2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2b
/conv2d/kernel/Regularizer/Square/ReadVariableOp/conv2d/kernel/Regularizer/Square/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2f
1conv2d_1/kernel/Regularizer/Square/ReadVariableOp1conv2d_1/kernel/Regularizer/Square/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2X
*layer_normalization/Reshape/ReadVariableOp*layer_normalization/Reshape/ReadVariableOp2\
,layer_normalization/Reshape_1/ReadVariableOp,layer_normalization/Reshape_1/ReadVariableOp2R
'photoreceptor_rods_reike/ReadVariableOp'photoreceptor_rods_reike/ReadVariableOp2V
)photoreceptor_rods_reike/ReadVariableOp_1)photoreceptor_rods_reike/ReadVariableOp_12V
)photoreceptor_rods_reike/ReadVariableOp_2)photoreceptor_rods_reike/ReadVariableOp_22V
)photoreceptor_rods_reike/ReadVariableOp_3)photoreceptor_rods_reike/ReadVariableOp_32V
)photoreceptor_rods_reike/ReadVariableOp_4)photoreceptor_rods_reike/ReadVariableOp_42V
)photoreceptor_rods_reike/ReadVariableOp_5)photoreceptor_rods_reike/ReadVariableOp_52V
)photoreceptor_rods_reike/ReadVariableOp_6)photoreceptor_rods_reike/ReadVariableOp_62V
)photoreceptor_rods_reike/ReadVariableOp_7)photoreceptor_rods_reike/ReadVariableOp_72V
)photoreceptor_rods_reike/ReadVariableOp_8)photoreceptor_rods_reike/ReadVariableOp_82V
)photoreceptor_rods_reike/ReadVariableOp_9)photoreceptor_rods_reike/ReadVariableOp_92d
0photoreceptor_rods_reike/StatefulPartitionedCall0photoreceptor_rods_reike/StatefulPartitionedCall2Z
+photoreceptor_rods_reike/mul/ReadVariableOp+photoreceptor_rods_reike/mul/ReadVariableOp2^
-photoreceptor_rods_reike/mul_1/ReadVariableOp-photoreceptor_rods_reike/mul_1/ReadVariableOp2^
-photoreceptor_rods_reike/mul_2/ReadVariableOp-photoreceptor_rods_reike/mul_2/ReadVariableOp2^
-photoreceptor_rods_reike/mul_3/ReadVariableOp-photoreceptor_rods_reike/mul_3/ReadVariableOp2^
-photoreceptor_rods_reike/mul_4/ReadVariableOp-photoreceptor_rods_reike/mul_4/ReadVariableOp2^
-photoreceptor_rods_reike/mul_5/ReadVariableOp-photoreceptor_rods_reike/mul_5/ReadVariableOp2^
-photoreceptor_rods_reike/mul_6/ReadVariableOp-photoreceptor_rods_reike/mul_6/ReadVariableOp2^
-photoreceptor_rods_reike/mul_7/ReadVariableOp-photoreceptor_rods_reike/mul_7/ReadVariableOp2^
-photoreceptor_rods_reike/mul_8/ReadVariableOp-photoreceptor_rods_reike/mul_8/ReadVariableOp2^
-photoreceptor_rods_reike/mul_9/ReadVariableOp-photoreceptor_rods_reike/mul_9/ReadVariableOp:X T
0
_output_shapes
:         ┤'
 
_user_specified_nameinputs
╜
╟
"__inference_signature_wrapper_3483
input_1
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20: 

unknown_21:x' 

unknown_22:x'$

unknown_23:		x

unknown_24:

unknown_25:	╨*

unknown_26:	╨*

unknown_27:	╨*

unknown_28:	╨*$

unknown_29:

unknown_30:

unknown_31:	╨

unknown_32:	╨

unknown_33:	╨

unknown_34:	╨$

unknown_35:

unknown_36:

unknown_37:Z

unknown_38:Z

unknown_39:Z

unknown_40:Z

unknown_41:Z

unknown_42:Z

unknown_43:Z

unknown_44:Z

unknown_45:Z%

unknown_46:%
identityИвStatefulPartitionedCall│
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46*<
Tin5
321*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         %*R
_read_only_resource_inputs4
20	
 !"#$%&'()*+,-./0*0
config_proto 

CPU

GPU2*0J 8В *(
f#R!
__inference__wrapped_model_1373o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         %`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*П
_input_shapes~
|:         ┤': : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:         ┤'
!
_user_specified_name	input_1
н
╙
4__inference_batch_normalization_1_layer_call_fn_4758

inputs
unknown:	╨
	unknown_0:	╨
	unknown_1:	╨
	unknown_2:	╨
identityИвStatefulPartitionedCallА
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ╨*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1538p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ╨`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ╨: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ╨
 
_user_specified_nameinputs
Т
f
J__inference_gaussian_noise_2_layer_call_and_return_conditional_losses_2108

inputs
identityV
IdentityIdentityinputs*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
й
╤
2__inference_batch_normalization_layer_call_fn_4562

inputs
unknown:	╨*
	unknown_0:	╨*
	unknown_1:	╨*
	unknown_2:	╨*
identityИвStatefulPartitionedCall■
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ╨**$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_1444p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ╨*`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ╨*: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ╨*
 
_user_specified_nameinputs
░
D
(__inference_reshape_2_layer_call_fn_4621

inputs
identity╣
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_reshape_2_layer_call_and_return_conditional_losses_1960h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ╨*:P L
(
_output_shapes
:         ╨*
 
_user_specified_nameinputs
├
_
C__inference_flatten_2_layer_call_and_return_conditional_losses_4908

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"    Z   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:         ZX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:         Z"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
ш
`
D__inference_activation_layer_call_and_return_conditional_losses_1979

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:         b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
Т
f
J__inference_gaussian_noise_1_layer_call_and_return_conditional_losses_2040

inputs
identityV
IdentityIdentityinputs*
T0*/
_output_shapes
:         	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         	:W S
/
_output_shapes
:         	
 
_user_specified_nameinputs
Ю
│
D__inference_CNNs_start_layer_call_and_return_conditional_losses_1923

inputs8
conv2d_readvariableop_resource:		x-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpв3CNNs_start/kernel/Regularizer/Square/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:		x*
dtype0▒
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
data_formatNCHW*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ф
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
data_formatNCHWЪ
3CNNs_start/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:		x*
dtype0Ь
$CNNs_start/kernel/Regularizer/SquareSquare;CNNs_start/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		x|
#CNNs_start/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             б
!CNNs_start/kernel/Regularizer/SumSum(CNNs_start/kernel/Regularizer/Square:y:0,CNNs_start/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#CNNs_start/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:г
!CNNs_start/kernel/Regularizer/mulMul,CNNs_start/kernel/Regularizer/mul/x:output:0*CNNs_start/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:         н
NoOpNoOp^BiasAdd/ReadVariableOp4^CNNs_start/kernel/Regularizer/Square/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         x': : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2j
3CNNs_start/kernel/Regularizer/Square/ReadVariableOp3CNNs_start/kernel/Regularizer/Square/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         x'
 
_user_specified_nameinputs
▄
░
M__inference_batch_normalization_layer_call_and_return_conditional_losses_1397

inputs0
!batchnorm_readvariableop_resource:	╨*4
%batchnorm_mul_readvariableop_resource:	╨*2
#batchnorm_readvariableop_1_resource:	╨*2
#batchnorm_readvariableop_2_resource:	╨*
identityИвbatchnorm/ReadVariableOpвbatchnorm/ReadVariableOp_1вbatchnorm/ReadVariableOp_2вbatchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:╨**
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:╨*Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:╨*
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:╨**
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:╨*d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         ╨*{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:╨**
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:╨*{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:╨**
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:╨*s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         ╨*c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:         ╨*║
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ╨*: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:         ╨*
 
_user_specified_nameinputs
Г
л
2__inference_layer_normalization_layer_call_fn_4468

inputs
unknown:x'
	unknown_0:x'
identityИвStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         x'*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_layer_normalization_layer_call_and_return_conditional_losses_1901w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         x'`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         x': : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         x'
 
_user_specified_nameinputs
Би
Д7
__inference__wrapped_model_1373
input_1R
@prfr_cnn2d_rods_photoreceptor_rods_reike_readvariableop_resource:V
Dprfr_cnn2d_rods_photoreceptor_rods_reike_mul_readvariableop_resource:T
Bprfr_cnn2d_rods_photoreceptor_rods_reike_readvariableop_1_resource:X
Fprfr_cnn2d_rods_photoreceptor_rods_reike_mul_1_readvariableop_resource:T
Bprfr_cnn2d_rods_photoreceptor_rods_reike_readvariableop_2_resource:X
Fprfr_cnn2d_rods_photoreceptor_rods_reike_mul_2_readvariableop_resource:T
Bprfr_cnn2d_rods_photoreceptor_rods_reike_readvariableop_3_resource:X
Fprfr_cnn2d_rods_photoreceptor_rods_reike_mul_3_readvariableop_resource:T
Bprfr_cnn2d_rods_photoreceptor_rods_reike_readvariableop_4_resource:X
Fprfr_cnn2d_rods_photoreceptor_rods_reike_mul_4_readvariableop_resource:T
Bprfr_cnn2d_rods_photoreceptor_rods_reike_readvariableop_5_resource:X
Fprfr_cnn2d_rods_photoreceptor_rods_reike_mul_5_readvariableop_resource:T
Bprfr_cnn2d_rods_photoreceptor_rods_reike_readvariableop_6_resource:X
Fprfr_cnn2d_rods_photoreceptor_rods_reike_mul_6_readvariableop_resource:T
Bprfr_cnn2d_rods_photoreceptor_rods_reike_readvariableop_7_resource:X
Fprfr_cnn2d_rods_photoreceptor_rods_reike_mul_7_readvariableop_resource:T
Bprfr_cnn2d_rods_photoreceptor_rods_reike_readvariableop_8_resource:X
Fprfr_cnn2d_rods_photoreceptor_rods_reike_mul_8_readvariableop_resource:T
Bprfr_cnn2d_rods_photoreceptor_rods_reike_readvariableop_9_resource:X
Fprfr_cnn2d_rods_photoreceptor_rods_reike_mul_9_readvariableop_resource:?
-prfr_cnn2d_rods_photoreceptor_rods_reike_1188:?
-prfr_cnn2d_rods_photoreceptor_rods_reike_1190:Y
Cprfr_cnn2d_rods_layer_normalization_reshape_readvariableop_resource:x'[
Eprfr_cnn2d_rods_layer_normalization_reshape_1_readvariableop_resource:x'S
9prfr_cnn2d_rods_cnns_start_conv2d_readvariableop_resource:		xH
:prfr_cnn2d_rods_cnns_start_biasadd_readvariableop_resource:T
Eprfr_cnn2d_rods_batch_normalization_batchnorm_readvariableop_resource:	╨*X
Iprfr_cnn2d_rods_batch_normalization_batchnorm_mul_readvariableop_resource:	╨*V
Gprfr_cnn2d_rods_batch_normalization_batchnorm_readvariableop_1_resource:	╨*V
Gprfr_cnn2d_rods_batch_normalization_batchnorm_readvariableop_2_resource:	╨*O
5prfr_cnn2d_rods_conv2d_conv2d_readvariableop_resource:D
6prfr_cnn2d_rods_conv2d_biasadd_readvariableop_resource:V
Gprfr_cnn2d_rods_batch_normalization_1_batchnorm_readvariableop_resource:	╨Z
Kprfr_cnn2d_rods_batch_normalization_1_batchnorm_mul_readvariableop_resource:	╨X
Iprfr_cnn2d_rods_batch_normalization_1_batchnorm_readvariableop_1_resource:	╨X
Iprfr_cnn2d_rods_batch_normalization_1_batchnorm_readvariableop_2_resource:	╨Q
7prfr_cnn2d_rods_conv2d_1_conv2d_readvariableop_resource:F
8prfr_cnn2d_rods_conv2d_1_biasadd_readvariableop_resource:U
Gprfr_cnn2d_rods_batch_normalization_2_batchnorm_readvariableop_resource:ZY
Kprfr_cnn2d_rods_batch_normalization_2_batchnorm_mul_readvariableop_resource:ZW
Iprfr_cnn2d_rods_batch_normalization_2_batchnorm_readvariableop_1_resource:ZW
Iprfr_cnn2d_rods_batch_normalization_2_batchnorm_readvariableop_2_resource:ZU
Gprfr_cnn2d_rods_batch_normalization_3_batchnorm_readvariableop_resource:ZY
Kprfr_cnn2d_rods_batch_normalization_3_batchnorm_mul_readvariableop_resource:ZW
Iprfr_cnn2d_rods_batch_normalization_3_batchnorm_readvariableop_1_resource:ZW
Iprfr_cnn2d_rods_batch_normalization_3_batchnorm_readvariableop_2_resource:ZF
4prfr_cnn2d_rods_dense_matmul_readvariableop_resource:Z%C
5prfr_cnn2d_rods_dense_biasadd_readvariableop_resource:%
identityИв1PRFR_CNN2D_RODS/CNNs_start/BiasAdd/ReadVariableOpв0PRFR_CNN2D_RODS/CNNs_start/Conv2D/ReadVariableOpв<PRFR_CNN2D_RODS/batch_normalization/batchnorm/ReadVariableOpв>PRFR_CNN2D_RODS/batch_normalization/batchnorm/ReadVariableOp_1в>PRFR_CNN2D_RODS/batch_normalization/batchnorm/ReadVariableOp_2в@PRFR_CNN2D_RODS/batch_normalization/batchnorm/mul/ReadVariableOpв>PRFR_CNN2D_RODS/batch_normalization_1/batchnorm/ReadVariableOpв@PRFR_CNN2D_RODS/batch_normalization_1/batchnorm/ReadVariableOp_1в@PRFR_CNN2D_RODS/batch_normalization_1/batchnorm/ReadVariableOp_2вBPRFR_CNN2D_RODS/batch_normalization_1/batchnorm/mul/ReadVariableOpв>PRFR_CNN2D_RODS/batch_normalization_2/batchnorm/ReadVariableOpв@PRFR_CNN2D_RODS/batch_normalization_2/batchnorm/ReadVariableOp_1в@PRFR_CNN2D_RODS/batch_normalization_2/batchnorm/ReadVariableOp_2вBPRFR_CNN2D_RODS/batch_normalization_2/batchnorm/mul/ReadVariableOpв>PRFR_CNN2D_RODS/batch_normalization_3/batchnorm/ReadVariableOpв@PRFR_CNN2D_RODS/batch_normalization_3/batchnorm/ReadVariableOp_1в@PRFR_CNN2D_RODS/batch_normalization_3/batchnorm/ReadVariableOp_2вBPRFR_CNN2D_RODS/batch_normalization_3/batchnorm/mul/ReadVariableOpв-PRFR_CNN2D_RODS/conv2d/BiasAdd/ReadVariableOpв,PRFR_CNN2D_RODS/conv2d/Conv2D/ReadVariableOpв/PRFR_CNN2D_RODS/conv2d_1/BiasAdd/ReadVariableOpв.PRFR_CNN2D_RODS/conv2d_1/Conv2D/ReadVariableOpв,PRFR_CNN2D_RODS/dense/BiasAdd/ReadVariableOpв+PRFR_CNN2D_RODS/dense/MatMul/ReadVariableOpв:PRFR_CNN2D_RODS/layer_normalization/Reshape/ReadVariableOpв<PRFR_CNN2D_RODS/layer_normalization/Reshape_1/ReadVariableOpв7PRFR_CNN2D_RODS/photoreceptor_rods_reike/ReadVariableOpв9PRFR_CNN2D_RODS/photoreceptor_rods_reike/ReadVariableOp_1в9PRFR_CNN2D_RODS/photoreceptor_rods_reike/ReadVariableOp_2в9PRFR_CNN2D_RODS/photoreceptor_rods_reike/ReadVariableOp_3в9PRFR_CNN2D_RODS/photoreceptor_rods_reike/ReadVariableOp_4в9PRFR_CNN2D_RODS/photoreceptor_rods_reike/ReadVariableOp_5в9PRFR_CNN2D_RODS/photoreceptor_rods_reike/ReadVariableOp_6в9PRFR_CNN2D_RODS/photoreceptor_rods_reike/ReadVariableOp_7в9PRFR_CNN2D_RODS/photoreceptor_rods_reike/ReadVariableOp_8в9PRFR_CNN2D_RODS/photoreceptor_rods_reike/ReadVariableOp_9в@PRFR_CNN2D_RODS/photoreceptor_rods_reike/StatefulPartitionedCallв;PRFR_CNN2D_RODS/photoreceptor_rods_reike/mul/ReadVariableOpв=PRFR_CNN2D_RODS/photoreceptor_rods_reike/mul_1/ReadVariableOpв=PRFR_CNN2D_RODS/photoreceptor_rods_reike/mul_2/ReadVariableOpв=PRFR_CNN2D_RODS/photoreceptor_rods_reike/mul_3/ReadVariableOpв=PRFR_CNN2D_RODS/photoreceptor_rods_reike/mul_4/ReadVariableOpв=PRFR_CNN2D_RODS/photoreceptor_rods_reike/mul_5/ReadVariableOpв=PRFR_CNN2D_RODS/photoreceptor_rods_reike/mul_6/ReadVariableOpв=PRFR_CNN2D_RODS/photoreceptor_rods_reike/mul_7/ReadVariableOpв=PRFR_CNN2D_RODS/photoreceptor_rods_reike/mul_8/ReadVariableOpв=PRFR_CNN2D_RODS/photoreceptor_rods_reike/mul_9/ReadVariableOpT
PRFR_CNN2D_RODS/reshape/ShapeShapeinput_1*
T0*
_output_shapes
:u
+PRFR_CNN2D_RODS/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-PRFR_CNN2D_RODS/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-PRFR_CNN2D_RODS/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╔
%PRFR_CNN2D_RODS/reshape/strided_sliceStridedSlice&PRFR_CNN2D_RODS/reshape/Shape:output:04PRFR_CNN2D_RODS/reshape/strided_slice/stack:output:06PRFR_CNN2D_RODS/reshape/strided_slice/stack_1:output:06PRFR_CNN2D_RODS/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
'PRFR_CNN2D_RODS/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :┤j
'PRFR_CNN2D_RODS/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :Т	я
%PRFR_CNN2D_RODS/reshape/Reshape/shapePack.PRFR_CNN2D_RODS/reshape/strided_slice:output:00PRFR_CNN2D_RODS/reshape/Reshape/shape/1:output:00PRFR_CNN2D_RODS/reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:Ы
PRFR_CNN2D_RODS/reshape/ReshapeReshapeinput_1.PRFR_CNN2D_RODS/reshape/Reshape/shape:output:0*
T0*-
_output_shapes
:         ┤Т	╕
7PRFR_CNN2D_RODS/photoreceptor_rods_reike/ReadVariableOpReadVariableOp@prfr_cnn2d_rods_photoreceptor_rods_reike_readvariableop_resource*
_output_shapes

:*
dtype0└
;PRFR_CNN2D_RODS/photoreceptor_rods_reike/mul/ReadVariableOpReadVariableOpDprfr_cnn2d_rods_photoreceptor_rods_reike_mul_readvariableop_resource*
_output_shapes

:*
dtype0т
,PRFR_CNN2D_RODS/photoreceptor_rods_reike/mulMul?PRFR_CNN2D_RODS/photoreceptor_rods_reike/ReadVariableOp:value:0CPRFR_CNN2D_RODS/photoreceptor_rods_reike/mul/ReadVariableOp:value:0*
T0*
_output_shapes

:╝
9PRFR_CNN2D_RODS/photoreceptor_rods_reike/ReadVariableOp_1ReadVariableOpBprfr_cnn2d_rods_photoreceptor_rods_reike_readvariableop_1_resource*
_output_shapes

:*
dtype0─
=PRFR_CNN2D_RODS/photoreceptor_rods_reike/mul_1/ReadVariableOpReadVariableOpFprfr_cnn2d_rods_photoreceptor_rods_reike_mul_1_readvariableop_resource*
_output_shapes

:*
dtype0ш
.PRFR_CNN2D_RODS/photoreceptor_rods_reike/mul_1MulAPRFR_CNN2D_RODS/photoreceptor_rods_reike/ReadVariableOp_1:value:0EPRFR_CNN2D_RODS/photoreceptor_rods_reike/mul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:╝
9PRFR_CNN2D_RODS/photoreceptor_rods_reike/ReadVariableOp_2ReadVariableOpBprfr_cnn2d_rods_photoreceptor_rods_reike_readvariableop_2_resource*
_output_shapes

:*
dtype0─
=PRFR_CNN2D_RODS/photoreceptor_rods_reike/mul_2/ReadVariableOpReadVariableOpFprfr_cnn2d_rods_photoreceptor_rods_reike_mul_2_readvariableop_resource*
_output_shapes

:*
dtype0ш
.PRFR_CNN2D_RODS/photoreceptor_rods_reike/mul_2MulAPRFR_CNN2D_RODS/photoreceptor_rods_reike/ReadVariableOp_2:value:0EPRFR_CNN2D_RODS/photoreceptor_rods_reike/mul_2/ReadVariableOp:value:0*
T0*
_output_shapes

:╝
9PRFR_CNN2D_RODS/photoreceptor_rods_reike/ReadVariableOp_3ReadVariableOpBprfr_cnn2d_rods_photoreceptor_rods_reike_readvariableop_3_resource*
_output_shapes

:*
dtype0─
=PRFR_CNN2D_RODS/photoreceptor_rods_reike/mul_3/ReadVariableOpReadVariableOpFprfr_cnn2d_rods_photoreceptor_rods_reike_mul_3_readvariableop_resource*
_output_shapes

:*
dtype0ш
.PRFR_CNN2D_RODS/photoreceptor_rods_reike/mul_3MulAPRFR_CNN2D_RODS/photoreceptor_rods_reike/ReadVariableOp_3:value:0EPRFR_CNN2D_RODS/photoreceptor_rods_reike/mul_3/ReadVariableOp:value:0*
T0*
_output_shapes

:╝
9PRFR_CNN2D_RODS/photoreceptor_rods_reike/ReadVariableOp_4ReadVariableOpBprfr_cnn2d_rods_photoreceptor_rods_reike_readvariableop_4_resource*
_output_shapes

:*
dtype0─
=PRFR_CNN2D_RODS/photoreceptor_rods_reike/mul_4/ReadVariableOpReadVariableOpFprfr_cnn2d_rods_photoreceptor_rods_reike_mul_4_readvariableop_resource*
_output_shapes

:*
dtype0ш
.PRFR_CNN2D_RODS/photoreceptor_rods_reike/mul_4MulAPRFR_CNN2D_RODS/photoreceptor_rods_reike/ReadVariableOp_4:value:0EPRFR_CNN2D_RODS/photoreceptor_rods_reike/mul_4/ReadVariableOp:value:0*
T0*
_output_shapes

:╝
9PRFR_CNN2D_RODS/photoreceptor_rods_reike/ReadVariableOp_5ReadVariableOpBprfr_cnn2d_rods_photoreceptor_rods_reike_readvariableop_5_resource*
_output_shapes

:*
dtype0─
=PRFR_CNN2D_RODS/photoreceptor_rods_reike/mul_5/ReadVariableOpReadVariableOpFprfr_cnn2d_rods_photoreceptor_rods_reike_mul_5_readvariableop_resource*
_output_shapes

:*
dtype0ш
.PRFR_CNN2D_RODS/photoreceptor_rods_reike/mul_5MulAPRFR_CNN2D_RODS/photoreceptor_rods_reike/ReadVariableOp_5:value:0EPRFR_CNN2D_RODS/photoreceptor_rods_reike/mul_5/ReadVariableOp:value:0*
T0*
_output_shapes

:╝
9PRFR_CNN2D_RODS/photoreceptor_rods_reike/ReadVariableOp_6ReadVariableOpBprfr_cnn2d_rods_photoreceptor_rods_reike_readvariableop_6_resource*
_output_shapes

:*
dtype0─
=PRFR_CNN2D_RODS/photoreceptor_rods_reike/mul_6/ReadVariableOpReadVariableOpFprfr_cnn2d_rods_photoreceptor_rods_reike_mul_6_readvariableop_resource*
_output_shapes

:*
dtype0ш
.PRFR_CNN2D_RODS/photoreceptor_rods_reike/mul_6MulAPRFR_CNN2D_RODS/photoreceptor_rods_reike/ReadVariableOp_6:value:0EPRFR_CNN2D_RODS/photoreceptor_rods_reike/mul_6/ReadVariableOp:value:0*
T0*
_output_shapes

:╝
9PRFR_CNN2D_RODS/photoreceptor_rods_reike/ReadVariableOp_7ReadVariableOpBprfr_cnn2d_rods_photoreceptor_rods_reike_readvariableop_7_resource*
_output_shapes

:*
dtype0─
=PRFR_CNN2D_RODS/photoreceptor_rods_reike/mul_7/ReadVariableOpReadVariableOpFprfr_cnn2d_rods_photoreceptor_rods_reike_mul_7_readvariableop_resource*
_output_shapes

:*
dtype0ш
.PRFR_CNN2D_RODS/photoreceptor_rods_reike/mul_7MulAPRFR_CNN2D_RODS/photoreceptor_rods_reike/ReadVariableOp_7:value:0EPRFR_CNN2D_RODS/photoreceptor_rods_reike/mul_7/ReadVariableOp:value:0*
T0*
_output_shapes

:╝
9PRFR_CNN2D_RODS/photoreceptor_rods_reike/ReadVariableOp_8ReadVariableOpBprfr_cnn2d_rods_photoreceptor_rods_reike_readvariableop_8_resource*
_output_shapes

:*
dtype0─
=PRFR_CNN2D_RODS/photoreceptor_rods_reike/mul_8/ReadVariableOpReadVariableOpFprfr_cnn2d_rods_photoreceptor_rods_reike_mul_8_readvariableop_resource*
_output_shapes

:*
dtype0ш
.PRFR_CNN2D_RODS/photoreceptor_rods_reike/mul_8MulAPRFR_CNN2D_RODS/photoreceptor_rods_reike/ReadVariableOp_8:value:0EPRFR_CNN2D_RODS/photoreceptor_rods_reike/mul_8/ReadVariableOp:value:0*
T0*
_output_shapes

:w
2PRFR_CNN2D_RODS/photoreceptor_rods_reike/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A╒
0PRFR_CNN2D_RODS/photoreceptor_rods_reike/truedivRealDiv2PRFR_CNN2D_RODS/photoreceptor_rods_reike/mul_8:z:0;PRFR_CNN2D_RODS/photoreceptor_rods_reike/truediv/y:output:0*
T0*
_output_shapes

:╝
9PRFR_CNN2D_RODS/photoreceptor_rods_reike/ReadVariableOp_9ReadVariableOpBprfr_cnn2d_rods_photoreceptor_rods_reike_readvariableop_9_resource*
_output_shapes

:*
dtype0─
=PRFR_CNN2D_RODS/photoreceptor_rods_reike/mul_9/ReadVariableOpReadVariableOpFprfr_cnn2d_rods_photoreceptor_rods_reike_mul_9_readvariableop_resource*
_output_shapes

:*
dtype0ш
.PRFR_CNN2D_RODS/photoreceptor_rods_reike/mul_9MulAPRFR_CNN2D_RODS/photoreceptor_rods_reike/ReadVariableOp_9:value:0EPRFR_CNN2D_RODS/photoreceptor_rods_reike/mul_9/ReadVariableOp:value:0*
T0*
_output_shapes

:р
@PRFR_CNN2D_RODS/photoreceptor_rods_reike/StatefulPartitionedCallStatefulPartitionedCall(PRFR_CNN2D_RODS/reshape/Reshape:output:00PRFR_CNN2D_RODS/photoreceptor_rods_reike/mul:z:02PRFR_CNN2D_RODS/photoreceptor_rods_reike/mul_1:z:02PRFR_CNN2D_RODS/photoreceptor_rods_reike/mul_2:z:0-prfr_cnn2d_rods_photoreceptor_rods_reike_11882PRFR_CNN2D_RODS/photoreceptor_rods_reike/mul_3:z:0-prfr_cnn2d_rods_photoreceptor_rods_reike_11902PRFR_CNN2D_RODS/photoreceptor_rods_reike/mul_4:z:02PRFR_CNN2D_RODS/photoreceptor_rods_reike/mul_5:z:02PRFR_CNN2D_RODS/photoreceptor_rods_reike/mul_6:z:02PRFR_CNN2D_RODS/photoreceptor_rods_reike/mul_7:z:04PRFR_CNN2D_RODS/photoreceptor_rods_reike/truediv:z:02PRFR_CNN2D_RODS/photoreceptor_rods_reike/mul_9:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         ┤Т	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *$
fR
__inference_riekeModel_1187Ш
PRFR_CNN2D_RODS/reshape_1/ShapeShapeIPRFR_CNN2D_RODS/photoreceptor_rods_reike/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:w
-PRFR_CNN2D_RODS/reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/PRFR_CNN2D_RODS/reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/PRFR_CNN2D_RODS/reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╙
'PRFR_CNN2D_RODS/reshape_1/strided_sliceStridedSlice(PRFR_CNN2D_RODS/reshape_1/Shape:output:06PRFR_CNN2D_RODS/reshape_1/strided_slice/stack:output:08PRFR_CNN2D_RODS/reshape_1/strided_slice/stack_1:output:08PRFR_CNN2D_RODS/reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
)PRFR_CNN2D_RODS/reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :┤k
)PRFR_CNN2D_RODS/reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :k
)PRFR_CNN2D_RODS/reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :'л
'PRFR_CNN2D_RODS/reshape_1/Reshape/shapePack0PRFR_CNN2D_RODS/reshape_1/strided_slice:output:02PRFR_CNN2D_RODS/reshape_1/Reshape/shape/1:output:02PRFR_CNN2D_RODS/reshape_1/Reshape/shape/2:output:02PRFR_CNN2D_RODS/reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:ф
!PRFR_CNN2D_RODS/reshape_1/ReshapeReshapeIPRFR_CNN2D_RODS/photoreceptor_rods_reike/StatefulPartitionedCall:output:00PRFR_CNN2D_RODS/reshape_1/Reshape/shape:output:0*
T0*0
_output_shapes
:         ┤'Х
<PRFR_CNN2D_RODS/tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"    <           Ч
>PRFR_CNN2D_RODS/tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                Ч
>PRFR_CNN2D_RODS/tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            ┤
6PRFR_CNN2D_RODS/tf.__operators__.getitem/strided_sliceStridedSlice*PRFR_CNN2D_RODS/reshape_1/Reshape:output:0EPRFR_CNN2D_RODS/tf.__operators__.getitem/strided_slice/stack:output:0GPRFR_CNN2D_RODS/tf.__operators__.getitem/strided_slice/stack_1:output:0GPRFR_CNN2D_RODS/tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         x'*

begin_mask*
end_maskЧ
BPRFR_CNN2D_RODS/layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         С
0PRFR_CNN2D_RODS/layer_normalization/moments/meanMean?PRFR_CNN2D_RODS/tf.__operators__.getitem/strided_slice:output:0KPRFR_CNN2D_RODS/layer_normalization/moments/mean/reduction_indices:output:0*
T0*/
_output_shapes
:         *
	keep_dims(╜
8PRFR_CNN2D_RODS/layer_normalization/moments/StopGradientStopGradient9PRFR_CNN2D_RODS/layer_normalization/moments/mean:output:0*
T0*/
_output_shapes
:         Р
=PRFR_CNN2D_RODS/layer_normalization/moments/SquaredDifferenceSquaredDifference?PRFR_CNN2D_RODS/tf.__operators__.getitem/strided_slice:output:0APRFR_CNN2D_RODS/layer_normalization/moments/StopGradient:output:0*
T0*/
_output_shapes
:         x'Ы
FPRFR_CNN2D_RODS/layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         Ы
4PRFR_CNN2D_RODS/layer_normalization/moments/varianceMeanAPRFR_CNN2D_RODS/layer_normalization/moments/SquaredDifference:z:0OPRFR_CNN2D_RODS/layer_normalization/moments/variance/reduction_indices:output:0*
T0*/
_output_shapes
:         *
	keep_dims(┬
:PRFR_CNN2D_RODS/layer_normalization/Reshape/ReadVariableOpReadVariableOpCprfr_cnn2d_rods_layer_normalization_reshape_readvariableop_resource*"
_output_shapes
:x'*
dtype0К
1PRFR_CNN2D_RODS/layer_normalization/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   x      '   ч
+PRFR_CNN2D_RODS/layer_normalization/ReshapeReshapeBPRFR_CNN2D_RODS/layer_normalization/Reshape/ReadVariableOp:value:0:PRFR_CNN2D_RODS/layer_normalization/Reshape/shape:output:0*
T0*&
_output_shapes
:x'╞
<PRFR_CNN2D_RODS/layer_normalization/Reshape_1/ReadVariableOpReadVariableOpEprfr_cnn2d_rods_layer_normalization_reshape_1_readvariableop_resource*"
_output_shapes
:x'*
dtype0М
3PRFR_CNN2D_RODS/layer_normalization/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   x      '   э
-PRFR_CNN2D_RODS/layer_normalization/Reshape_1ReshapeDPRFR_CNN2D_RODS/layer_normalization/Reshape_1/ReadVariableOp:value:0<PRFR_CNN2D_RODS/layer_normalization/Reshape_1/shape:output:0*
T0*&
_output_shapes
:x'x
3PRFR_CNN2D_RODS/layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Х┐╓3ё
1PRFR_CNN2D_RODS/layer_normalization/batchnorm/addAddV2=PRFR_CNN2D_RODS/layer_normalization/moments/variance:output:0<PRFR_CNN2D_RODS/layer_normalization/batchnorm/add/y:output:0*
T0*/
_output_shapes
:         н
3PRFR_CNN2D_RODS/layer_normalization/batchnorm/RsqrtRsqrt5PRFR_CNN2D_RODS/layer_normalization/batchnorm/add:z:0*
T0*/
_output_shapes
:         с
1PRFR_CNN2D_RODS/layer_normalization/batchnorm/mulMul7PRFR_CNN2D_RODS/layer_normalization/batchnorm/Rsqrt:y:04PRFR_CNN2D_RODS/layer_normalization/Reshape:output:0*
T0*/
_output_shapes
:         x'ь
3PRFR_CNN2D_RODS/layer_normalization/batchnorm/mul_1Mul?PRFR_CNN2D_RODS/tf.__operators__.getitem/strided_slice:output:05PRFR_CNN2D_RODS/layer_normalization/batchnorm/mul:z:0*
T0*/
_output_shapes
:         x'ц
3PRFR_CNN2D_RODS/layer_normalization/batchnorm/mul_2Mul9PRFR_CNN2D_RODS/layer_normalization/moments/mean:output:05PRFR_CNN2D_RODS/layer_normalization/batchnorm/mul:z:0*
T0*/
_output_shapes
:         x'у
1PRFR_CNN2D_RODS/layer_normalization/batchnorm/subSub6PRFR_CNN2D_RODS/layer_normalization/Reshape_1:output:07PRFR_CNN2D_RODS/layer_normalization/batchnorm/mul_2:z:0*
T0*/
_output_shapes
:         x'ц
3PRFR_CNN2D_RODS/layer_normalization/batchnorm/add_1AddV27PRFR_CNN2D_RODS/layer_normalization/batchnorm/mul_1:z:05PRFR_CNN2D_RODS/layer_normalization/batchnorm/sub:z:0*
T0*/
_output_shapes
:         x'▓
0PRFR_CNN2D_RODS/CNNs_start/Conv2D/ReadVariableOpReadVariableOp9prfr_cnn2d_rods_cnns_start_conv2d_readvariableop_resource*&
_output_shapes
:		x*
dtype0Ш
!PRFR_CNN2D_RODS/CNNs_start/Conv2DConv2D7PRFR_CNN2D_RODS/layer_normalization/batchnorm/add_1:z:08PRFR_CNN2D_RODS/CNNs_start/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
data_formatNCHW*
paddingVALID*
strides
и
1PRFR_CNN2D_RODS/CNNs_start/BiasAdd/ReadVariableOpReadVariableOp:prfr_cnn2d_rods_cnns_start_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0х
"PRFR_CNN2D_RODS/CNNs_start/BiasAddBiasAdd*PRFR_CNN2D_RODS/CNNs_start/Conv2D:output:09PRFR_CNN2D_RODS/CNNs_start/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
data_formatNCHWn
PRFR_CNN2D_RODS/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"    P  ▓
PRFR_CNN2D_RODS/flatten/ReshapeReshape+PRFR_CNN2D_RODS/CNNs_start/BiasAdd:output:0&PRFR_CNN2D_RODS/flatten/Const:output:0*
T0*(
_output_shapes
:         ╨*┐
<PRFR_CNN2D_RODS/batch_normalization/batchnorm/ReadVariableOpReadVariableOpEprfr_cnn2d_rods_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:╨**
dtype0x
3PRFR_CNN2D_RODS/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:ф
1PRFR_CNN2D_RODS/batch_normalization/batchnorm/addAddV2DPRFR_CNN2D_RODS/batch_normalization/batchnorm/ReadVariableOp:value:0<PRFR_CNN2D_RODS/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:╨*Щ
3PRFR_CNN2D_RODS/batch_normalization/batchnorm/RsqrtRsqrt5PRFR_CNN2D_RODS/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:╨*╟
@PRFR_CNN2D_RODS/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOpIprfr_cnn2d_rods_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:╨**
dtype0с
1PRFR_CNN2D_RODS/batch_normalization/batchnorm/mulMul7PRFR_CNN2D_RODS/batch_normalization/batchnorm/Rsqrt:y:0HPRFR_CNN2D_RODS/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:╨*╬
3PRFR_CNN2D_RODS/batch_normalization/batchnorm/mul_1Mul(PRFR_CNN2D_RODS/flatten/Reshape:output:05PRFR_CNN2D_RODS/batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:         ╨*├
>PRFR_CNN2D_RODS/batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOpGprfr_cnn2d_rods_batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes	
:╨**
dtype0▀
3PRFR_CNN2D_RODS/batch_normalization/batchnorm/mul_2MulFPRFR_CNN2D_RODS/batch_normalization/batchnorm/ReadVariableOp_1:value:05PRFR_CNN2D_RODS/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:╨*├
>PRFR_CNN2D_RODS/batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOpGprfr_cnn2d_rods_batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes	
:╨**
dtype0▀
1PRFR_CNN2D_RODS/batch_normalization/batchnorm/subSubFPRFR_CNN2D_RODS/batch_normalization/batchnorm/ReadVariableOp_2:value:07PRFR_CNN2D_RODS/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:╨*▀
3PRFR_CNN2D_RODS/batch_normalization/batchnorm/add_1AddV27PRFR_CNN2D_RODS/batch_normalization/batchnorm/mul_1:z:05PRFR_CNN2D_RODS/batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:         ╨*Ж
PRFR_CNN2D_RODS/reshape_2/ShapeShape7PRFR_CNN2D_RODS/batch_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:w
-PRFR_CNN2D_RODS/reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/PRFR_CNN2D_RODS/reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/PRFR_CNN2D_RODS/reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╙
'PRFR_CNN2D_RODS/reshape_2/strided_sliceStridedSlice(PRFR_CNN2D_RODS/reshape_2/Shape:output:06PRFR_CNN2D_RODS/reshape_2/strided_slice/stack:output:08PRFR_CNN2D_RODS/reshape_2/strided_slice/stack_1:output:08PRFR_CNN2D_RODS/reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
)PRFR_CNN2D_RODS/reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :k
)PRFR_CNN2D_RODS/reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :k
)PRFR_CNN2D_RODS/reshape_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :л
'PRFR_CNN2D_RODS/reshape_2/Reshape/shapePack0PRFR_CNN2D_RODS/reshape_2/strided_slice:output:02PRFR_CNN2D_RODS/reshape_2/Reshape/shape/1:output:02PRFR_CNN2D_RODS/reshape_2/Reshape/shape/2:output:02PRFR_CNN2D_RODS/reshape_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:╤
!PRFR_CNN2D_RODS/reshape_2/ReshapeReshape7PRFR_CNN2D_RODS/batch_normalization/batchnorm/add_1:z:00PRFR_CNN2D_RODS/reshape_2/Reshape/shape:output:0*
T0*/
_output_shapes
:         р
%PRFR_CNN2D_RODS/max_pooling2d/MaxPoolMaxPool*PRFR_CNN2D_RODS/reshape_2/Reshape:output:0*/
_output_shapes
:         *
data_formatNCHW*
ksize
*
paddingVALID*
strides
С
PRFR_CNN2D_RODS/activation/ReluRelu.PRFR_CNN2D_RODS/max_pooling2d/MaxPool:output:0*
T0*/
_output_shapes
:         к
,PRFR_CNN2D_RODS/conv2d/Conv2D/ReadVariableOpReadVariableOp5prfr_cnn2d_rods_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ж
PRFR_CNN2D_RODS/conv2d/Conv2DConv2D-PRFR_CNN2D_RODS/activation/Relu:activations:04PRFR_CNN2D_RODS/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         	*
data_formatNCHW*
paddingVALID*
strides
а
-PRFR_CNN2D_RODS/conv2d/BiasAdd/ReadVariableOpReadVariableOp6prfr_cnn2d_rods_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0┘
PRFR_CNN2D_RODS/conv2d/BiasAddBiasAdd&PRFR_CNN2D_RODS/conv2d/Conv2D:output:05PRFR_CNN2D_RODS/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         	*
data_formatNCHWp
PRFR_CNN2D_RODS/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"    ╨  ▓
!PRFR_CNN2D_RODS/flatten_1/ReshapeReshape'PRFR_CNN2D_RODS/conv2d/BiasAdd:output:0(PRFR_CNN2D_RODS/flatten_1/Const:output:0*
T0*(
_output_shapes
:         ╨├
>PRFR_CNN2D_RODS/batch_normalization_1/batchnorm/ReadVariableOpReadVariableOpGprfr_cnn2d_rods_batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:╨*
dtype0z
5PRFR_CNN2D_RODS/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:ъ
3PRFR_CNN2D_RODS/batch_normalization_1/batchnorm/addAddV2FPRFR_CNN2D_RODS/batch_normalization_1/batchnorm/ReadVariableOp:value:0>PRFR_CNN2D_RODS/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:╨Э
5PRFR_CNN2D_RODS/batch_normalization_1/batchnorm/RsqrtRsqrt7PRFR_CNN2D_RODS/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:╨╦
BPRFR_CNN2D_RODS/batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpKprfr_cnn2d_rods_batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:╨*
dtype0ч
3PRFR_CNN2D_RODS/batch_normalization_1/batchnorm/mulMul9PRFR_CNN2D_RODS/batch_normalization_1/batchnorm/Rsqrt:y:0JPRFR_CNN2D_RODS/batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:╨╘
5PRFR_CNN2D_RODS/batch_normalization_1/batchnorm/mul_1Mul*PRFR_CNN2D_RODS/flatten_1/Reshape:output:07PRFR_CNN2D_RODS/batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:         ╨╟
@PRFR_CNN2D_RODS/batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOpIprfr_cnn2d_rods_batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes	
:╨*
dtype0х
5PRFR_CNN2D_RODS/batch_normalization_1/batchnorm/mul_2MulHPRFR_CNN2D_RODS/batch_normalization_1/batchnorm/ReadVariableOp_1:value:07PRFR_CNN2D_RODS/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:╨╟
@PRFR_CNN2D_RODS/batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOpIprfr_cnn2d_rods_batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes	
:╨*
dtype0х
3PRFR_CNN2D_RODS/batch_normalization_1/batchnorm/subSubHPRFR_CNN2D_RODS/batch_normalization_1/batchnorm/ReadVariableOp_2:value:09PRFR_CNN2D_RODS/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:╨х
5PRFR_CNN2D_RODS/batch_normalization_1/batchnorm/add_1AddV29PRFR_CNN2D_RODS/batch_normalization_1/batchnorm/mul_1:z:07PRFR_CNN2D_RODS/batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:         ╨И
PRFR_CNN2D_RODS/reshape_3/ShapeShape9PRFR_CNN2D_RODS/batch_normalization_1/batchnorm/add_1:z:0*
T0*
_output_shapes
:w
-PRFR_CNN2D_RODS/reshape_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/PRFR_CNN2D_RODS/reshape_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/PRFR_CNN2D_RODS/reshape_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╙
'PRFR_CNN2D_RODS/reshape_3/strided_sliceStridedSlice(PRFR_CNN2D_RODS/reshape_3/Shape:output:06PRFR_CNN2D_RODS/reshape_3/strided_slice/stack:output:08PRFR_CNN2D_RODS/reshape_3/strided_slice/stack_1:output:08PRFR_CNN2D_RODS/reshape_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
)PRFR_CNN2D_RODS/reshape_3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :k
)PRFR_CNN2D_RODS/reshape_3/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :k
)PRFR_CNN2D_RODS/reshape_3/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :	л
'PRFR_CNN2D_RODS/reshape_3/Reshape/shapePack0PRFR_CNN2D_RODS/reshape_3/strided_slice:output:02PRFR_CNN2D_RODS/reshape_3/Reshape/shape/1:output:02PRFR_CNN2D_RODS/reshape_3/Reshape/shape/2:output:02PRFR_CNN2D_RODS/reshape_3/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:╙
!PRFR_CNN2D_RODS/reshape_3/ReshapeReshape9PRFR_CNN2D_RODS/batch_normalization_1/batchnorm/add_1:z:00PRFR_CNN2D_RODS/reshape_3/Reshape/shape:output:0*
T0*/
_output_shapes
:         	П
!PRFR_CNN2D_RODS/activation_1/ReluRelu*PRFR_CNN2D_RODS/reshape_3/Reshape:output:0*
T0*/
_output_shapes
:         	о
.PRFR_CNN2D_RODS/conv2d_1/Conv2D/ReadVariableOpReadVariableOp7prfr_cnn2d_rods_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0М
PRFR_CNN2D_RODS/conv2d_1/Conv2DConv2D/PRFR_CNN2D_RODS/activation_1/Relu:activations:06PRFR_CNN2D_RODS/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
data_formatNCHW*
paddingVALID*
strides
д
/PRFR_CNN2D_RODS/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp8prfr_cnn2d_rods_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0▀
 PRFR_CNN2D_RODS/conv2d_1/BiasAddBiasAdd(PRFR_CNN2D_RODS/conv2d_1/Conv2D:output:07PRFR_CNN2D_RODS/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
data_formatNCHWp
PRFR_CNN2D_RODS/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"    Z   │
!PRFR_CNN2D_RODS/flatten_2/ReshapeReshape)PRFR_CNN2D_RODS/conv2d_1/BiasAdd:output:0(PRFR_CNN2D_RODS/flatten_2/Const:output:0*
T0*'
_output_shapes
:         Z┬
>PRFR_CNN2D_RODS/batch_normalization_2/batchnorm/ReadVariableOpReadVariableOpGprfr_cnn2d_rods_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:Z*
dtype0z
5PRFR_CNN2D_RODS/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:щ
3PRFR_CNN2D_RODS/batch_normalization_2/batchnorm/addAddV2FPRFR_CNN2D_RODS/batch_normalization_2/batchnorm/ReadVariableOp:value:0>PRFR_CNN2D_RODS/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:ZЬ
5PRFR_CNN2D_RODS/batch_normalization_2/batchnorm/RsqrtRsqrt7PRFR_CNN2D_RODS/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:Z╩
BPRFR_CNN2D_RODS/batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpKprfr_cnn2d_rods_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:Z*
dtype0ц
3PRFR_CNN2D_RODS/batch_normalization_2/batchnorm/mulMul9PRFR_CNN2D_RODS/batch_normalization_2/batchnorm/Rsqrt:y:0JPRFR_CNN2D_RODS/batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Z╙
5PRFR_CNN2D_RODS/batch_normalization_2/batchnorm/mul_1Mul*PRFR_CNN2D_RODS/flatten_2/Reshape:output:07PRFR_CNN2D_RODS/batch_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:         Z╞
@PRFR_CNN2D_RODS/batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOpIprfr_cnn2d_rods_batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes
:Z*
dtype0ф
5PRFR_CNN2D_RODS/batch_normalization_2/batchnorm/mul_2MulHPRFR_CNN2D_RODS/batch_normalization_2/batchnorm/ReadVariableOp_1:value:07PRFR_CNN2D_RODS/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:Z╞
@PRFR_CNN2D_RODS/batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOpIprfr_cnn2d_rods_batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes
:Z*
dtype0ф
3PRFR_CNN2D_RODS/batch_normalization_2/batchnorm/subSubHPRFR_CNN2D_RODS/batch_normalization_2/batchnorm/ReadVariableOp_2:value:09PRFR_CNN2D_RODS/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Zф
5PRFR_CNN2D_RODS/batch_normalization_2/batchnorm/add_1AddV29PRFR_CNN2D_RODS/batch_normalization_2/batchnorm/mul_1:z:07PRFR_CNN2D_RODS/batch_normalization_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:         ZИ
PRFR_CNN2D_RODS/reshape_4/ShapeShape9PRFR_CNN2D_RODS/batch_normalization_2/batchnorm/add_1:z:0*
T0*
_output_shapes
:w
-PRFR_CNN2D_RODS/reshape_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/PRFR_CNN2D_RODS/reshape_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/PRFR_CNN2D_RODS/reshape_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╙
'PRFR_CNN2D_RODS/reshape_4/strided_sliceStridedSlice(PRFR_CNN2D_RODS/reshape_4/Shape:output:06PRFR_CNN2D_RODS/reshape_4/strided_slice/stack:output:08PRFR_CNN2D_RODS/reshape_4/strided_slice/stack_1:output:08PRFR_CNN2D_RODS/reshape_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
)PRFR_CNN2D_RODS/reshape_4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :k
)PRFR_CNN2D_RODS/reshape_4/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :k
)PRFR_CNN2D_RODS/reshape_4/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :л
'PRFR_CNN2D_RODS/reshape_4/Reshape/shapePack0PRFR_CNN2D_RODS/reshape_4/strided_slice:output:02PRFR_CNN2D_RODS/reshape_4/Reshape/shape/1:output:02PRFR_CNN2D_RODS/reshape_4/Reshape/shape/2:output:02PRFR_CNN2D_RODS/reshape_4/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:╙
!PRFR_CNN2D_RODS/reshape_4/ReshapeReshape9PRFR_CNN2D_RODS/batch_normalization_2/batchnorm/add_1:z:00PRFR_CNN2D_RODS/reshape_4/Reshape/shape:output:0*
T0*/
_output_shapes
:         П
!PRFR_CNN2D_RODS/activation_2/ReluRelu*PRFR_CNN2D_RODS/reshape_4/Reshape:output:0*
T0*/
_output_shapes
:         p
PRFR_CNN2D_RODS/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"    Z   ╣
!PRFR_CNN2D_RODS/flatten_3/ReshapeReshape/PRFR_CNN2D_RODS/activation_2/Relu:activations:0(PRFR_CNN2D_RODS/flatten_3/Const:output:0*
T0*'
_output_shapes
:         Z┬
>PRFR_CNN2D_RODS/batch_normalization_3/batchnorm/ReadVariableOpReadVariableOpGprfr_cnn2d_rods_batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:Z*
dtype0z
5PRFR_CNN2D_RODS/batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:щ
3PRFR_CNN2D_RODS/batch_normalization_3/batchnorm/addAddV2FPRFR_CNN2D_RODS/batch_normalization_3/batchnorm/ReadVariableOp:value:0>PRFR_CNN2D_RODS/batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
:ZЬ
5PRFR_CNN2D_RODS/batch_normalization_3/batchnorm/RsqrtRsqrt7PRFR_CNN2D_RODS/batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:Z╩
BPRFR_CNN2D_RODS/batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOpKprfr_cnn2d_rods_batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:Z*
dtype0ц
3PRFR_CNN2D_RODS/batch_normalization_3/batchnorm/mulMul9PRFR_CNN2D_RODS/batch_normalization_3/batchnorm/Rsqrt:y:0JPRFR_CNN2D_RODS/batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Z╙
5PRFR_CNN2D_RODS/batch_normalization_3/batchnorm/mul_1Mul*PRFR_CNN2D_RODS/flatten_3/Reshape:output:07PRFR_CNN2D_RODS/batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:         Z╞
@PRFR_CNN2D_RODS/batch_normalization_3/batchnorm/ReadVariableOp_1ReadVariableOpIprfr_cnn2d_rods_batch_normalization_3_batchnorm_readvariableop_1_resource*
_output_shapes
:Z*
dtype0ф
5PRFR_CNN2D_RODS/batch_normalization_3/batchnorm/mul_2MulHPRFR_CNN2D_RODS/batch_normalization_3/batchnorm/ReadVariableOp_1:value:07PRFR_CNN2D_RODS/batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:Z╞
@PRFR_CNN2D_RODS/batch_normalization_3/batchnorm/ReadVariableOp_2ReadVariableOpIprfr_cnn2d_rods_batch_normalization_3_batchnorm_readvariableop_2_resource*
_output_shapes
:Z*
dtype0ф
3PRFR_CNN2D_RODS/batch_normalization_3/batchnorm/subSubHPRFR_CNN2D_RODS/batch_normalization_3/batchnorm/ReadVariableOp_2:value:09PRFR_CNN2D_RODS/batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Zф
5PRFR_CNN2D_RODS/batch_normalization_3/batchnorm/add_1AddV29PRFR_CNN2D_RODS/batch_normalization_3/batchnorm/mul_1:z:07PRFR_CNN2D_RODS/batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:         Zа
+PRFR_CNN2D_RODS/dense/MatMul/ReadVariableOpReadVariableOp4prfr_cnn2d_rods_dense_matmul_readvariableop_resource*
_output_shapes

:Z%*
dtype0╚
PRFR_CNN2D_RODS/dense/MatMulMatMul9PRFR_CNN2D_RODS/batch_normalization_3/batchnorm/add_1:z:03PRFR_CNN2D_RODS/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         %Ю
,PRFR_CNN2D_RODS/dense/BiasAdd/ReadVariableOpReadVariableOp5prfr_cnn2d_rods_dense_biasadd_readvariableop_resource*
_output_shapes
:%*
dtype0╕
PRFR_CNN2D_RODS/dense/BiasAddBiasAdd&PRFR_CNN2D_RODS/dense/MatMul:product:04PRFR_CNN2D_RODS/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         %О
-PRFR_CNN2D_RODS/dense/ActivityRegularizer/AbsAbs&PRFR_CNN2D_RODS/dense/BiasAdd:output:0*
T0*'
_output_shapes
:         %А
/PRFR_CNN2D_RODS/dense/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ┬
-PRFR_CNN2D_RODS/dense/ActivityRegularizer/SumSum1PRFR_CNN2D_RODS/dense/ActivityRegularizer/Abs:y:08PRFR_CNN2D_RODS/dense/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: t
/PRFR_CNN2D_RODS/dense/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:╟
-PRFR_CNN2D_RODS/dense/ActivityRegularizer/mulMul8PRFR_CNN2D_RODS/dense/ActivityRegularizer/mul/x:output:06PRFR_CNN2D_RODS/dense/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: Е
/PRFR_CNN2D_RODS/dense/ActivityRegularizer/ShapeShape&PRFR_CNN2D_RODS/dense/BiasAdd:output:0*
T0*
_output_shapes
:З
=PRFR_CNN2D_RODS/dense/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Й
?PRFR_CNN2D_RODS/dense/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Й
?PRFR_CNN2D_RODS/dense/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:г
7PRFR_CNN2D_RODS/dense/ActivityRegularizer/strided_sliceStridedSlice8PRFR_CNN2D_RODS/dense/ActivityRegularizer/Shape:output:0FPRFR_CNN2D_RODS/dense/ActivityRegularizer/strided_slice/stack:output:0HPRFR_CNN2D_RODS/dense/ActivityRegularizer/strided_slice/stack_1:output:0HPRFR_CNN2D_RODS/dense/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskи
.PRFR_CNN2D_RODS/dense/ActivityRegularizer/CastCast@PRFR_CNN2D_RODS/dense/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ─
1PRFR_CNN2D_RODS/dense/ActivityRegularizer/truedivRealDiv1PRFR_CNN2D_RODS/dense/ActivityRegularizer/mul:z:02PRFR_CNN2D_RODS/dense/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: Л
%PRFR_CNN2D_RODS/activation_3/SoftplusSoftplus&PRFR_CNN2D_RODS/dense/BiasAdd:output:0*
T0*'
_output_shapes
:         %В
IdentityIdentity3PRFR_CNN2D_RODS/activation_3/Softplus:activations:0^NoOp*
T0*'
_output_shapes
:         %З
NoOpNoOp2^PRFR_CNN2D_RODS/CNNs_start/BiasAdd/ReadVariableOp1^PRFR_CNN2D_RODS/CNNs_start/Conv2D/ReadVariableOp=^PRFR_CNN2D_RODS/batch_normalization/batchnorm/ReadVariableOp?^PRFR_CNN2D_RODS/batch_normalization/batchnorm/ReadVariableOp_1?^PRFR_CNN2D_RODS/batch_normalization/batchnorm/ReadVariableOp_2A^PRFR_CNN2D_RODS/batch_normalization/batchnorm/mul/ReadVariableOp?^PRFR_CNN2D_RODS/batch_normalization_1/batchnorm/ReadVariableOpA^PRFR_CNN2D_RODS/batch_normalization_1/batchnorm/ReadVariableOp_1A^PRFR_CNN2D_RODS/batch_normalization_1/batchnorm/ReadVariableOp_2C^PRFR_CNN2D_RODS/batch_normalization_1/batchnorm/mul/ReadVariableOp?^PRFR_CNN2D_RODS/batch_normalization_2/batchnorm/ReadVariableOpA^PRFR_CNN2D_RODS/batch_normalization_2/batchnorm/ReadVariableOp_1A^PRFR_CNN2D_RODS/batch_normalization_2/batchnorm/ReadVariableOp_2C^PRFR_CNN2D_RODS/batch_normalization_2/batchnorm/mul/ReadVariableOp?^PRFR_CNN2D_RODS/batch_normalization_3/batchnorm/ReadVariableOpA^PRFR_CNN2D_RODS/batch_normalization_3/batchnorm/ReadVariableOp_1A^PRFR_CNN2D_RODS/batch_normalization_3/batchnorm/ReadVariableOp_2C^PRFR_CNN2D_RODS/batch_normalization_3/batchnorm/mul/ReadVariableOp.^PRFR_CNN2D_RODS/conv2d/BiasAdd/ReadVariableOp-^PRFR_CNN2D_RODS/conv2d/Conv2D/ReadVariableOp0^PRFR_CNN2D_RODS/conv2d_1/BiasAdd/ReadVariableOp/^PRFR_CNN2D_RODS/conv2d_1/Conv2D/ReadVariableOp-^PRFR_CNN2D_RODS/dense/BiasAdd/ReadVariableOp,^PRFR_CNN2D_RODS/dense/MatMul/ReadVariableOp;^PRFR_CNN2D_RODS/layer_normalization/Reshape/ReadVariableOp=^PRFR_CNN2D_RODS/layer_normalization/Reshape_1/ReadVariableOp8^PRFR_CNN2D_RODS/photoreceptor_rods_reike/ReadVariableOp:^PRFR_CNN2D_RODS/photoreceptor_rods_reike/ReadVariableOp_1:^PRFR_CNN2D_RODS/photoreceptor_rods_reike/ReadVariableOp_2:^PRFR_CNN2D_RODS/photoreceptor_rods_reike/ReadVariableOp_3:^PRFR_CNN2D_RODS/photoreceptor_rods_reike/ReadVariableOp_4:^PRFR_CNN2D_RODS/photoreceptor_rods_reike/ReadVariableOp_5:^PRFR_CNN2D_RODS/photoreceptor_rods_reike/ReadVariableOp_6:^PRFR_CNN2D_RODS/photoreceptor_rods_reike/ReadVariableOp_7:^PRFR_CNN2D_RODS/photoreceptor_rods_reike/ReadVariableOp_8:^PRFR_CNN2D_RODS/photoreceptor_rods_reike/ReadVariableOp_9A^PRFR_CNN2D_RODS/photoreceptor_rods_reike/StatefulPartitionedCall<^PRFR_CNN2D_RODS/photoreceptor_rods_reike/mul/ReadVariableOp>^PRFR_CNN2D_RODS/photoreceptor_rods_reike/mul_1/ReadVariableOp>^PRFR_CNN2D_RODS/photoreceptor_rods_reike/mul_2/ReadVariableOp>^PRFR_CNN2D_RODS/photoreceptor_rods_reike/mul_3/ReadVariableOp>^PRFR_CNN2D_RODS/photoreceptor_rods_reike/mul_4/ReadVariableOp>^PRFR_CNN2D_RODS/photoreceptor_rods_reike/mul_5/ReadVariableOp>^PRFR_CNN2D_RODS/photoreceptor_rods_reike/mul_6/ReadVariableOp>^PRFR_CNN2D_RODS/photoreceptor_rods_reike/mul_7/ReadVariableOp>^PRFR_CNN2D_RODS/photoreceptor_rods_reike/mul_8/ReadVariableOp>^PRFR_CNN2D_RODS/photoreceptor_rods_reike/mul_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*П
_input_shapes~
|:         ┤': : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2f
1PRFR_CNN2D_RODS/CNNs_start/BiasAdd/ReadVariableOp1PRFR_CNN2D_RODS/CNNs_start/BiasAdd/ReadVariableOp2d
0PRFR_CNN2D_RODS/CNNs_start/Conv2D/ReadVariableOp0PRFR_CNN2D_RODS/CNNs_start/Conv2D/ReadVariableOp2|
<PRFR_CNN2D_RODS/batch_normalization/batchnorm/ReadVariableOp<PRFR_CNN2D_RODS/batch_normalization/batchnorm/ReadVariableOp2А
>PRFR_CNN2D_RODS/batch_normalization/batchnorm/ReadVariableOp_1>PRFR_CNN2D_RODS/batch_normalization/batchnorm/ReadVariableOp_12А
>PRFR_CNN2D_RODS/batch_normalization/batchnorm/ReadVariableOp_2>PRFR_CNN2D_RODS/batch_normalization/batchnorm/ReadVariableOp_22Д
@PRFR_CNN2D_RODS/batch_normalization/batchnorm/mul/ReadVariableOp@PRFR_CNN2D_RODS/batch_normalization/batchnorm/mul/ReadVariableOp2А
>PRFR_CNN2D_RODS/batch_normalization_1/batchnorm/ReadVariableOp>PRFR_CNN2D_RODS/batch_normalization_1/batchnorm/ReadVariableOp2Д
@PRFR_CNN2D_RODS/batch_normalization_1/batchnorm/ReadVariableOp_1@PRFR_CNN2D_RODS/batch_normalization_1/batchnorm/ReadVariableOp_12Д
@PRFR_CNN2D_RODS/batch_normalization_1/batchnorm/ReadVariableOp_2@PRFR_CNN2D_RODS/batch_normalization_1/batchnorm/ReadVariableOp_22И
BPRFR_CNN2D_RODS/batch_normalization_1/batchnorm/mul/ReadVariableOpBPRFR_CNN2D_RODS/batch_normalization_1/batchnorm/mul/ReadVariableOp2А
>PRFR_CNN2D_RODS/batch_normalization_2/batchnorm/ReadVariableOp>PRFR_CNN2D_RODS/batch_normalization_2/batchnorm/ReadVariableOp2Д
@PRFR_CNN2D_RODS/batch_normalization_2/batchnorm/ReadVariableOp_1@PRFR_CNN2D_RODS/batch_normalization_2/batchnorm/ReadVariableOp_12Д
@PRFR_CNN2D_RODS/batch_normalization_2/batchnorm/ReadVariableOp_2@PRFR_CNN2D_RODS/batch_normalization_2/batchnorm/ReadVariableOp_22И
BPRFR_CNN2D_RODS/batch_normalization_2/batchnorm/mul/ReadVariableOpBPRFR_CNN2D_RODS/batch_normalization_2/batchnorm/mul/ReadVariableOp2А
>PRFR_CNN2D_RODS/batch_normalization_3/batchnorm/ReadVariableOp>PRFR_CNN2D_RODS/batch_normalization_3/batchnorm/ReadVariableOp2Д
@PRFR_CNN2D_RODS/batch_normalization_3/batchnorm/ReadVariableOp_1@PRFR_CNN2D_RODS/batch_normalization_3/batchnorm/ReadVariableOp_12Д
@PRFR_CNN2D_RODS/batch_normalization_3/batchnorm/ReadVariableOp_2@PRFR_CNN2D_RODS/batch_normalization_3/batchnorm/ReadVariableOp_22И
BPRFR_CNN2D_RODS/batch_normalization_3/batchnorm/mul/ReadVariableOpBPRFR_CNN2D_RODS/batch_normalization_3/batchnorm/mul/ReadVariableOp2^
-PRFR_CNN2D_RODS/conv2d/BiasAdd/ReadVariableOp-PRFR_CNN2D_RODS/conv2d/BiasAdd/ReadVariableOp2\
,PRFR_CNN2D_RODS/conv2d/Conv2D/ReadVariableOp,PRFR_CNN2D_RODS/conv2d/Conv2D/ReadVariableOp2b
/PRFR_CNN2D_RODS/conv2d_1/BiasAdd/ReadVariableOp/PRFR_CNN2D_RODS/conv2d_1/BiasAdd/ReadVariableOp2`
.PRFR_CNN2D_RODS/conv2d_1/Conv2D/ReadVariableOp.PRFR_CNN2D_RODS/conv2d_1/Conv2D/ReadVariableOp2\
,PRFR_CNN2D_RODS/dense/BiasAdd/ReadVariableOp,PRFR_CNN2D_RODS/dense/BiasAdd/ReadVariableOp2Z
+PRFR_CNN2D_RODS/dense/MatMul/ReadVariableOp+PRFR_CNN2D_RODS/dense/MatMul/ReadVariableOp2x
:PRFR_CNN2D_RODS/layer_normalization/Reshape/ReadVariableOp:PRFR_CNN2D_RODS/layer_normalization/Reshape/ReadVariableOp2|
<PRFR_CNN2D_RODS/layer_normalization/Reshape_1/ReadVariableOp<PRFR_CNN2D_RODS/layer_normalization/Reshape_1/ReadVariableOp2r
7PRFR_CNN2D_RODS/photoreceptor_rods_reike/ReadVariableOp7PRFR_CNN2D_RODS/photoreceptor_rods_reike/ReadVariableOp2v
9PRFR_CNN2D_RODS/photoreceptor_rods_reike/ReadVariableOp_19PRFR_CNN2D_RODS/photoreceptor_rods_reike/ReadVariableOp_12v
9PRFR_CNN2D_RODS/photoreceptor_rods_reike/ReadVariableOp_29PRFR_CNN2D_RODS/photoreceptor_rods_reike/ReadVariableOp_22v
9PRFR_CNN2D_RODS/photoreceptor_rods_reike/ReadVariableOp_39PRFR_CNN2D_RODS/photoreceptor_rods_reike/ReadVariableOp_32v
9PRFR_CNN2D_RODS/photoreceptor_rods_reike/ReadVariableOp_49PRFR_CNN2D_RODS/photoreceptor_rods_reike/ReadVariableOp_42v
9PRFR_CNN2D_RODS/photoreceptor_rods_reike/ReadVariableOp_59PRFR_CNN2D_RODS/photoreceptor_rods_reike/ReadVariableOp_52v
9PRFR_CNN2D_RODS/photoreceptor_rods_reike/ReadVariableOp_69PRFR_CNN2D_RODS/photoreceptor_rods_reike/ReadVariableOp_62v
9PRFR_CNN2D_RODS/photoreceptor_rods_reike/ReadVariableOp_79PRFR_CNN2D_RODS/photoreceptor_rods_reike/ReadVariableOp_72v
9PRFR_CNN2D_RODS/photoreceptor_rods_reike/ReadVariableOp_89PRFR_CNN2D_RODS/photoreceptor_rods_reike/ReadVariableOp_82v
9PRFR_CNN2D_RODS/photoreceptor_rods_reike/ReadVariableOp_99PRFR_CNN2D_RODS/photoreceptor_rods_reike/ReadVariableOp_92Д
@PRFR_CNN2D_RODS/photoreceptor_rods_reike/StatefulPartitionedCall@PRFR_CNN2D_RODS/photoreceptor_rods_reike/StatefulPartitionedCall2z
;PRFR_CNN2D_RODS/photoreceptor_rods_reike/mul/ReadVariableOp;PRFR_CNN2D_RODS/photoreceptor_rods_reike/mul/ReadVariableOp2~
=PRFR_CNN2D_RODS/photoreceptor_rods_reike/mul_1/ReadVariableOp=PRFR_CNN2D_RODS/photoreceptor_rods_reike/mul_1/ReadVariableOp2~
=PRFR_CNN2D_RODS/photoreceptor_rods_reike/mul_2/ReadVariableOp=PRFR_CNN2D_RODS/photoreceptor_rods_reike/mul_2/ReadVariableOp2~
=PRFR_CNN2D_RODS/photoreceptor_rods_reike/mul_3/ReadVariableOp=PRFR_CNN2D_RODS/photoreceptor_rods_reike/mul_3/ReadVariableOp2~
=PRFR_CNN2D_RODS/photoreceptor_rods_reike/mul_4/ReadVariableOp=PRFR_CNN2D_RODS/photoreceptor_rods_reike/mul_4/ReadVariableOp2~
=PRFR_CNN2D_RODS/photoreceptor_rods_reike/mul_5/ReadVariableOp=PRFR_CNN2D_RODS/photoreceptor_rods_reike/mul_5/ReadVariableOp2~
=PRFR_CNN2D_RODS/photoreceptor_rods_reike/mul_6/ReadVariableOp=PRFR_CNN2D_RODS/photoreceptor_rods_reike/mul_6/ReadVariableOp2~
=PRFR_CNN2D_RODS/photoreceptor_rods_reike/mul_7/ReadVariableOp=PRFR_CNN2D_RODS/photoreceptor_rods_reike/mul_7/ReadVariableOp2~
=PRFR_CNN2D_RODS/photoreceptor_rods_reike/mul_8/ReadVariableOp=PRFR_CNN2D_RODS/photoreceptor_rods_reike/mul_8/ReadVariableOp2~
=PRFR_CNN2D_RODS/photoreceptor_rods_reike/mul_9/ReadVariableOp=PRFR_CNN2D_RODS/photoreceptor_rods_reike/mul_9/ReadVariableOp:Y U
0
_output_shapes
:         ┤'
!
_user_specified_name	input_1
Р%
ш
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_5133

inputs5
'assignmovingavg_readvariableop_resource:Z7
)assignmovingavg_1_readvariableop_resource:Z3
%batchnorm_mul_readvariableop_resource:Z/
!batchnorm_readvariableop_resource:Z
identityИвAssignMovingAvgвAssignMovingAvg/ReadVariableOpвAssignMovingAvg_1в AssignMovingAvg_1/ReadVariableOpвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:Z*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:ZЗ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:         Zl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ю
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:Z*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:Z*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:Z*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<В
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:Z*
dtype0Б
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:Zx
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:Zм
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<Ж
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:Z*
dtype0З
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:Z~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:Z┤
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:ZP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:Z~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:Z*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Zc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:         Zh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:Zv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:Z*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Zr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:         Zb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:         Zъ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         Z: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:         Z
 
_user_specified_nameinputs
м
B
&__inference_flatten_layer_call_fn_4530

inputs
identity░
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ╨** 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_1935a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         ╨*"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
─
G
+__inference_activation_2_layer_call_fn_5037

inputs
identity╝
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_activation_2_layer_call_and_return_conditional_losses_2115h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
н%
ь
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4812

inputs6
'assignmovingavg_readvariableop_resource:	╨8
)assignmovingavg_1_readvariableop_resource:	╨4
%batchnorm_mul_readvariableop_resource:	╨0
!batchnorm_readvariableop_resource:	╨
identityИвAssignMovingAvgвAssignMovingAvg/ReadVariableOpвAssignMovingAvg_1в AssignMovingAvg_1/ReadVariableOpвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: А
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	╨*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	╨И
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:         ╨l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Я
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	╨*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:╨*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:╨*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<Г
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:╨*
dtype0В
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:╨y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:╨м
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<З
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:╨*
dtype0И
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:╨
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:╨┤
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:╨Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:╨
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:╨*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:╨d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         ╨i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:╨w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:╨*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:╨s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         ╨c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:         ╨ъ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ╨: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:         ╨
 
_user_specified_nameinputs
╠
_
C__inference_reshape_3_layer_call_and_return_conditional_losses_4831

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :	й
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:         	`
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:         	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ╨:P L
(
_output_shapes
:         ╨
 
_user_specified_nameinputs
М
└
C__inference_dense_layer_call_and_return_all_conditional_losses_5159

inputs
unknown:Z%
	unknown_0:%
identity

identity_1ИвStatefulPartitionedCall╫
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         %*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_2150в
PartitionedCallPartitionedCall StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *4
f/R-
+__inference_dense_activity_regularizer_1726o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         %X

Identity_1IdentityPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         Z: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         Z
 
_user_specified_nameinputs
╩
_
C__inference_reshape_4_layer_call_and_return_conditional_losses_5007

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :й
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:         `
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         Z:O K
'
_output_shapes
:         Z
 
_user_specified_nameinputs
°
B
+__inference_dense_activity_regularizer_1726
x
identity0
AbsAbsx*
T0*
_output_shapes
:6
RankRankAbs:y:0*
T0*
_output_shapes
: M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :n
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*#
_output_shapes
:         D
SumSumAbs:y:0range:output:0*
T0*
_output_shapes
: J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:I
mulMulmul/x:output:0Sum:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
::; 7

_output_shapes
:

_user_specified_namex
▐
▓
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4778

inputs0
!batchnorm_readvariableop_resource:	╨4
%batchnorm_mul_readvariableop_resource:	╨2
#batchnorm_readvariableop_1_resource:	╨2
#batchnorm_readvariableop_2_resource:	╨
identityИвbatchnorm/ReadVariableOpвbatchnorm/ReadVariableOp_1вbatchnorm/ReadVariableOp_2вbatchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:╨*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:╨Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:╨
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:╨*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:╨d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         ╨{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:╨*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:╨{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:╨*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:╨s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         ╨c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:         ╨║
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ╨: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:         ╨
 
_user_specified_nameinputs
а
┘
while_cond_1039
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_placeholder_4
while_placeholder_5
while_placeholder_6
while_less_range_1_limit2
.while_while_cond_1039___redundant_placeholder02
.while_while_cond_1039___redundant_placeholder12
.while_while_cond_1039___redundant_placeholder22
.while_while_cond_1039___redundant_placeholder32
.while_while_cond_1039___redundant_placeholder42
.while_while_cond_1039___redundant_placeholder52
.while_while_cond_1039___redundant_placeholder62
.while_while_cond_1039___redundant_placeholder72
.while_while_cond_1039___redundant_placeholder82
.while_while_cond_1039___redundant_placeholder93
/while_while_cond_1039___redundant_placeholder103
/while_while_cond_1039___redundant_placeholder113
/while_while_cond_1039___redundant_placeholder12
while_identity
`

while/LessLesswhile_placeholderwhile_less_range_1_limit*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*╖
_input_shapesе
в: : : : :         Т	:         Т	:         Т	:         Т	:         Т	: :::::::::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         Т	:.*
(
_output_shapes
:         Т	:.*
(
_output_shapes
:         Т	:.*
(
_output_shapes
:         Т	:.*
(
_output_shapes
:         Т	:	

_output_shapes
: :


_output_shapes
::

_output_shapes
::

_output_shapes
::

_output_shapes
::

_output_shapes
::

_output_shapes
::

_output_shapes
::

_output_shapes
::

_output_shapes
::

_output_shapes
::

_output_shapes
::

_output_shapes
:
д	
g
H__inference_gaussian_noise_layer_call_and_return_conditional_losses_2456

inputs
identityИ;
ShapeShapeinputs*
T0*
_output_shapes
:W
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=С
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*/
_output_shapes
:         *
dtype0Ю
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*/
_output_shapes
:         Д
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*/
_output_shapes
:         a
addAddV2inputsrandom_normal:z:0*
T0*/
_output_shapes
:         W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
ы

]
A__inference_reshape_layer_call_and_return_conditional_losses_4330

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskR
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :┤R
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :Т	П
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:j
ReshapeReshapeinputsReshape/shape:output:0*
T0*-
_output_shapes
:         ┤Т	^
IdentityIdentityReshape:output:0*
T0*-
_output_shapes
:         ┤Т	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ┤':X T
0
_output_shapes
:         ┤'
 
_user_specified_nameinputs
├
_
C__inference_flatten_2_layer_call_and_return_conditional_losses_2077

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"    Z   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:         ZX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:         Z"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
┘
_
C__inference_reshape_1_layer_call_and_return_conditional_losses_4459

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskR
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :┤Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :'й
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:m
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:         ┤'a
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:         ┤'"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:         ┤Т	:U Q
-
_output_shapes
:         ┤Т	
 
_user_specified_nameinputs
╠
о
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_4954

inputs/
!batchnorm_readvariableop_resource:Z3
%batchnorm_mul_readvariableop_resource:Z1
#batchnorm_readvariableop_1_resource:Z1
#batchnorm_readvariableop_2_resource:Z
identityИвbatchnorm/ReadVariableOpвbatchnorm/ReadVariableOp_1вbatchnorm/ReadVariableOp_2вbatchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:Z*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:ZP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:Z~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:Z*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Zc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:         Zz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:Z*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:Zz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:Z*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Zr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:         Zb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:         Z║
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         Z: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:         Z
 
_user_specified_nameinputs
о
D
(__inference_flatten_3_layer_call_fn_5047

inputs
identity▒
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         Z* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_flatten_3_layer_call_and_return_conditional_losses_2123`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         Z"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
є
╥
.__inference_PRFR_CNN2D_RODS_layer_call_fn_3585

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20: 

unknown_21:x' 

unknown_22:x'$

unknown_23:		x

unknown_24:

unknown_25:	╨*

unknown_26:	╨*

unknown_27:	╨*

unknown_28:	╨*$

unknown_29:

unknown_30:

unknown_31:	╨

unknown_32:	╨

unknown_33:	╨

unknown_34:	╨$

unknown_35:

unknown_36:

unknown_37:Z

unknown_38:Z

unknown_39:Z

unknown_40:Z

unknown_41:Z

unknown_42:Z

unknown_43:Z

unknown_44:Z

unknown_45:Z%

unknown_46:%
identityИвStatefulPartitionedCall▀
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46*<
Tin5
321*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         %: *R
_read_only_resource_inputs4
20	
 !"#$%&'()*+,-./0*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_PRFR_CNN2D_RODS_layer_call_and_return_conditional_losses_2197o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         %`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*П
_input_shapes~
|:         ┤': : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         ┤'
 
_user_specified_nameinputs
║
c
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_1966

inputs
identityЮ
MaxPoolMaxPoolinputs*/
_output_shapes
:         *
data_formatNCHW*
ksize
*
paddingVALID*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
е
╧
4__inference_batch_normalization_3_layer_call_fn_5079

inputs
unknown:Z
	unknown_0:Z
	unknown_1:Z
	unknown_2:Z
identityИвStatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         Z*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1702o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         Z`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         Z: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         Z
 
_user_specified_nameinputs
ж	
i
J__inference_gaussian_noise_2_layer_call_and_return_conditional_losses_2356

inputs
identityИ;
ShapeShapeinputs*
T0*
_output_shapes
:W
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=С
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*/
_output_shapes
:         *
dtype0Ю
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*/
_output_shapes
:         Д
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*/
_output_shapes
:         a
addAddV2inputsrandom_normal:z:0*
T0*/
_output_shapes
:         W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
н%
ь
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1538

inputs6
'assignmovingavg_readvariableop_resource:	╨8
)assignmovingavg_1_readvariableop_resource:	╨4
%batchnorm_mul_readvariableop_resource:	╨0
!batchnorm_readvariableop_resource:	╨
identityИвAssignMovingAvgвAssignMovingAvg/ReadVariableOpвAssignMovingAvg_1в AssignMovingAvg_1/ReadVariableOpвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: А
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	╨*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	╨И
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:         ╨l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Я
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	╨*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:╨*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:╨*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<Г
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:╨*
dtype0В
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:╨y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:╨м
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<З
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:╨*
dtype0И
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:╨
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:╨┤
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:╨Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:╨
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:╨*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:╨d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         ╨i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:╨w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:╨*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:╨s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         ╨c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:         ╨ъ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ╨: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:         ╨
 
_user_specified_nameinputs
▐
л
@__inference_conv2d_layer_call_and_return_conditional_losses_1997

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpв/conv2d/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0▒
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         	*
data_formatNCHW*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ф
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         	*
data_formatNCHWЦ
/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ф
 conv2d/kernel/Regularizer/SquareSquare7conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:x
conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             Х
conv2d/kernel/Regularizer/SumSum$conv2d/kernel/Regularizer/Square:y:0(conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:Ч
conv2d/kernel/Regularizer/mulMul(conv2d/kernel/Regularizer/mul/x:output:0&conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:         	й
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp0^conv2d/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2b
/conv2d/kernel/Regularizer/Square/ReadVariableOp/conv2d/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
Г
ў
7__inference_photoreceptor_rods_reike_layer_call_fn_4379

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:
identityИвStatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         ┤Т	*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_photoreceptor_rods_reike_layer_call_and_return_conditional_losses_1809u
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*-
_output_shapes
:         ┤Т	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:         ┤Т	: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
-
_output_shapes
:         ┤Т	
 
_user_specified_nameinputs
з
╧
4__inference_batch_normalization_3_layer_call_fn_5066

inputs
unknown:Z
	unknown_0:Z
	unknown_1:Z
	unknown_2:Z
identityИвStatefulPartitionedCallБ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         Z*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1655o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         Z`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         Z: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         Z
 
_user_specified_nameinputs
ш
`
D__inference_activation_layer_call_and_return_conditional_losses_4690

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:         b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
■
п
B__inference_conv2d_1_layer_call_and_return_conditional_losses_4897

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpв1conv2d_1/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0▒
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
data_formatNCHW*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ф
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
data_formatNCHWШ
1conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ш
"conv2d_1/kernel/Regularizer/SquareSquare9conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
!conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             Ы
conv2d_1/kernel/Regularizer/SumSum&conv2d_1/kernel/Regularizer/Square:y:0*conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:Э
conv2d_1/kernel/Regularizer/mulMul*conv2d_1/kernel/Regularizer/mul/x:output:0(conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:         л
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp2^conv2d_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2f
1conv2d_1/kernel/Regularizer/Square/ReadVariableOp1conv2d_1/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:         	
 
_user_specified_nameinputs
з
╧
4__inference_batch_normalization_2_layer_call_fn_4921

inputs
unknown:Z
	unknown_0:Z
	unknown_1:Z
	unknown_2:Z
identityИвStatefulPartitionedCallБ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         Z*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1573o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         Z`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         Z: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         Z
 
_user_specified_nameinputs
├
]
A__inference_flatten_layer_call_and_return_conditional_losses_1935

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"    P  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         ╨*Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         ╨*"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
о
D
(__inference_reshape_4_layer_call_fn_4993

inputs
identity╣
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_reshape_4_layer_call_and_return_conditional_losses_2102h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         Z:O K
'
_output_shapes
:         Z
 
_user_specified_nameinputs
Р%
ш
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_4988

inputs5
'assignmovingavg_readvariableop_resource:Z7
)assignmovingavg_1_readvariableop_resource:Z3
%batchnorm_mul_readvariableop_resource:Z/
!batchnorm_readvariableop_resource:Z
identityИвAssignMovingAvgвAssignMovingAvg/ReadVariableOpвAssignMovingAvg_1в AssignMovingAvg_1/ReadVariableOpвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:Z*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:ZЗ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:         Zl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ю
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:Z*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:Z*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:Z*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<В
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:Z*
dtype0Б
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:Zx
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:Zм
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<Ж
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:Z*
dtype0З
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:Z~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:Z┤
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:ZP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:Z~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:Z*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Zc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:         Zh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:Zv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:Z*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Zr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:         Zb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:         Zъ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         Z: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:         Z
 
_user_specified_nameinputs
ж	
i
J__inference_gaussian_noise_2_layer_call_and_return_conditional_losses_5032

inputs
identityИ;
ShapeShapeinputs*
T0*
_output_shapes
:W
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=С
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*/
_output_shapes
:         *
dtype0Ю
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*/
_output_shapes
:         Д
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*/
_output_shapes
:         a
addAddV2inputsrandom_normal:z:0*
T0*/
_output_shapes
:         W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
х
Ъ
%__inference_conv2d_layer_call_fn_4705

inputs!
unknown:
	unknown_0:
identityИвStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_1997w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
╠
_
C__inference_reshape_2_layer_call_and_return_conditional_losses_1960

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :й
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:         `
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ╨*:P L
(
_output_shapes
:         ╨*
 
_user_specified_nameinputs
Ъ
f
-__inference_gaussian_noise_layer_call_fn_4665

inputs
identityИвStatefulPartitionedCall╬
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_gaussian_noise_layer_call_and_return_conditional_losses_2456w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
▐
Р
M__inference_layer_normalization_layer_call_and_return_conditional_losses_1901

inputs5
reshape_readvariableop_resource:x'7
!reshape_1_readvariableop_resource:x'
identityИвReshape/ReadVariableOpвReshape_1/ReadVariableOps
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         Р
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*/
_output_shapes
:         *
	keep_dims(u
moments/StopGradientStopGradientmoments/mean:output:0*
T0*/
_output_shapes
:         П
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*/
_output_shapes
:         x'w
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         п
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*/
_output_shapes
:         *
	keep_dims(z
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*"
_output_shapes
:x'*
dtype0f
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   x      '   {
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*&
_output_shapes
:x'~
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*"
_output_shapes
:x'*
dtype0h
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   x      '   Б
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:x'T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Х┐╓3Е
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*/
_output_shapes
:         e
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*/
_output_shapes
:         u
batchnorm/mulMulbatchnorm/Rsqrt:y:0Reshape:output:0*
T0*/
_output_shapes
:         x'k
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*/
_output_shapes
:         x'z
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*/
_output_shapes
:         x'w
batchnorm/subSubReshape_1:output:0batchnorm/mul_2:z:0*
T0*/
_output_shapes
:         x'z
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*/
_output_shapes
:         x'j
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*/
_output_shapes
:         x'z
NoOpNoOp^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         x': : 20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:W S
/
_output_shapes
:         x'
 
_user_specified_nameinputs
е
╧
4__inference_batch_normalization_2_layer_call_fn_4934

inputs
unknown:Z
	unknown_0:Z
	unknown_1:Z
	unknown_2:Z
identityИвStatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         Z*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1620o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         Z`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         Z: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         Z
 
_user_specified_nameinputs
╠
о
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1655

inputs/
!batchnorm_readvariableop_resource:Z3
%batchnorm_mul_readvariableop_resource:Z1
#batchnorm_readvariableop_1_resource:Z1
#batchnorm_readvariableop_2_resource:Z
identityИвbatchnorm/ReadVariableOpвbatchnorm/ReadVariableOp_1вbatchnorm/ReadVariableOp_2вbatchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:Z*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:ZP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:Z~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:Z*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Zc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:         Zz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:Z*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:Zz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:Z*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Zr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:         Zb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:         Z║
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         Z: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:         Z
 
_user_specified_nameinputs
╠
K
/__inference_gaussian_noise_1_layer_call_fn_4836

inputs
identity└
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_gaussian_noise_1_layer_call_and_return_conditional_losses_2040h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         	:W S
/
_output_shapes
:         	
 
_user_specified_nameinputs
░
D
(__inference_flatten_1_layer_call_fn_4726

inputs
identity▓
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ╨* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_flatten_1_layer_call_and_return_conditional_losses_2009a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         ╨"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         	:W S
/
_output_shapes
:         	
 
_user_specified_nameinputs
ъ
b
F__inference_activation_2_layer_call_and_return_conditional_losses_5042

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:         b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
ю
╙
.__inference_PRFR_CNN2D_RODS_layer_call_fn_3028
input_1
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20: 

unknown_21:x' 

unknown_22:x'$

unknown_23:		x

unknown_24:

unknown_25:	╨*

unknown_26:	╨*

unknown_27:	╨*

unknown_28:	╨*$

unknown_29:

unknown_30:

unknown_31:	╨

unknown_32:	╨

unknown_33:	╨

unknown_34:	╨$

unknown_35:

unknown_36:

unknown_37:Z

unknown_38:Z

unknown_39:Z

unknown_40:Z

unknown_41:Z

unknown_42:Z

unknown_43:Z

unknown_44:Z

unknown_45:Z%

unknown_46:%
identityИвStatefulPartitionedCall╪
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46*<
Tin5
321*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         %: *J
_read_only_resource_inputs,
*(	
 #$%&)*-./0*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_PRFR_CNN2D_RODS_layer_call_and_return_conditional_losses_2826o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         %`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*П
_input_shapes~
|:         ┤': : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:         ┤'
!
_user_specified_name	input_1
││
м
I__inference_PRFR_CNN2D_RODS_layer_call_and_return_conditional_losses_2826

inputs/
photoreceptor_rods_reike_2666:/
photoreceptor_rods_reike_2668:/
photoreceptor_rods_reike_2670:/
photoreceptor_rods_reike_2672:/
photoreceptor_rods_reike_2674:/
photoreceptor_rods_reike_2676:/
photoreceptor_rods_reike_2678:/
photoreceptor_rods_reike_2680:/
photoreceptor_rods_reike_2682:/
photoreceptor_rods_reike_2684:/
photoreceptor_rods_reike_2686:/
photoreceptor_rods_reike_2688:/
photoreceptor_rods_reike_2690:/
photoreceptor_rods_reike_2692:/
photoreceptor_rods_reike_2694:/
photoreceptor_rods_reike_2696:/
photoreceptor_rods_reike_2698:/
photoreceptor_rods_reike_2700:/
photoreceptor_rods_reike_2702:/
photoreceptor_rods_reike_2704:/
photoreceptor_rods_reike_2706:/
photoreceptor_rods_reike_2708:.
layer_normalization_2716:x'.
layer_normalization_2718:x')
cnns_start_2721:		x
cnns_start_2723:'
batch_normalization_2727:	╨*'
batch_normalization_2729:	╨*'
batch_normalization_2731:	╨*'
batch_normalization_2733:	╨*%
conv2d_2740:
conv2d_2742:)
batch_normalization_1_2746:	╨)
batch_normalization_1_2748:	╨)
batch_normalization_1_2750:	╨)
batch_normalization_1_2752:	╨'
conv2d_1_2758:
conv2d_1_2760:(
batch_normalization_2_2764:Z(
batch_normalization_2_2766:Z(
batch_normalization_2_2768:Z(
batch_normalization_2_2770:Z(
batch_normalization_3_2777:Z(
batch_normalization_3_2779:Z(
batch_normalization_3_2781:Z(
batch_normalization_3_2783:Z

dense_2786:Z%

dense_2788:%
identity

identity_1Ив"CNNs_start/StatefulPartitionedCallв3CNNs_start/kernel/Regularizer/Square/ReadVariableOpв+batch_normalization/StatefulPartitionedCallв-batch_normalization_1/StatefulPartitionedCallв-batch_normalization_2/StatefulPartitionedCallв-batch_normalization_3/StatefulPartitionedCallвconv2d/StatefulPartitionedCallв/conv2d/kernel/Regularizer/Square/ReadVariableOpв conv2d_1/StatefulPartitionedCallв1conv2d_1/kernel/Regularizer/Square/ReadVariableOpвdense/StatefulPartitionedCallв.dense/kernel/Regularizer/Square/ReadVariableOpв&gaussian_noise/StatefulPartitionedCallв(gaussian_noise_1/StatefulPartitionedCallв(gaussian_noise_2/StatefulPartitionedCallв+layer_normalization/StatefulPartitionedCallв0photoreceptor_rods_reike/StatefulPartitionedCall╜
reshape/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         ┤Т	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_reshape_layer_call_and_return_conditional_losses_1746с
0photoreceptor_rods_reike/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0photoreceptor_rods_reike_2666photoreceptor_rods_reike_2668photoreceptor_rods_reike_2670photoreceptor_rods_reike_2672photoreceptor_rods_reike_2674photoreceptor_rods_reike_2676photoreceptor_rods_reike_2678photoreceptor_rods_reike_2680photoreceptor_rods_reike_2682photoreceptor_rods_reike_2684photoreceptor_rods_reike_2686photoreceptor_rods_reike_2688photoreceptor_rods_reike_2690photoreceptor_rods_reike_2692photoreceptor_rods_reike_2694photoreceptor_rods_reike_2696photoreceptor_rods_reike_2698photoreceptor_rods_reike_2700photoreceptor_rods_reike_2702photoreceptor_rods_reike_2704photoreceptor_rods_reike_2706photoreceptor_rods_reike_2708*"
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         ┤Т	*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_photoreceptor_rods_reike_layer_call_and_return_conditional_losses_1809ў
reshape_1/PartitionedCallPartitionedCall9photoreceptor_rods_reike/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ┤'* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_reshape_1_layer_call_and_return_conditional_losses_1869Е
,tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"    <           З
.tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                З
.tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            ь
&tf.__operators__.getitem/strided_sliceStridedSlice"reshape_1/PartitionedCall:output:05tf.__operators__.getitem/strided_slice/stack:output:07tf.__operators__.getitem/strided_slice/stack_1:output:07tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         x'*

begin_mask*
end_mask╩
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCall/tf.__operators__.getitem/strided_slice:output:0layer_normalization_2716layer_normalization_2718*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         x'*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_layer_normalization_layer_call_and_return_conditional_losses_1901л
"CNNs_start/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0cnns_start_2721cnns_start_2723*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_CNNs_start_layer_call_and_return_conditional_losses_1923▌
flatten/PartitionedCallPartitionedCall+CNNs_start/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ╨** 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_1935ъ
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0batch_normalization_2727batch_normalization_2729batch_normalization_2731batch_normalization_2733*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ╨**$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_1444ё
reshape_2/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_reshape_2_layer_call_and_return_conditional_losses_1960ч
max_pooling2d/PartitionedCallPartitionedCall"reshape_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_1966¤
&gaussian_noise/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_gaussian_noise_layer_call_and_return_conditional_losses_2456ю
activation/PartitionedCallPartitionedCall/gaussian_noise/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_activation_layer_call_and_return_conditional_losses_1979К
conv2d/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0conv2d_2740conv2d_2742*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_1997▌
flatten_1/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ╨* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_flatten_1_layer_call_and_return_conditional_losses_2009°
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0batch_normalization_1_2746batch_normalization_1_2748batch_normalization_1_2750batch_normalization_1_2752*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ╨*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1538є
reshape_3/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_reshape_3_layer_call_and_return_conditional_losses_2034ж
(gaussian_noise_1/StatefulPartitionedCallStatefulPartitionedCall"reshape_3/PartitionedCall:output:0'^gaussian_noise/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_gaussian_noise_1_layer_call_and_return_conditional_losses_2406Ї
activation_1/PartitionedCallPartitionedCall1gaussian_noise_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_activation_1_layer_call_and_return_conditional_losses_2047Ф
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0conv2d_1_2758conv2d_1_2760*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_2065▐
flatten_2/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         Z* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_flatten_2_layer_call_and_return_conditional_losses_2077ў
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0batch_normalization_2_2764batch_normalization_2_2766batch_normalization_2_2768batch_normalization_2_2770*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         Z*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1620є
reshape_4/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_reshape_4_layer_call_and_return_conditional_losses_2102и
(gaussian_noise_2/StatefulPartitionedCallStatefulPartitionedCall"reshape_4/PartitionedCall:output:0)^gaussian_noise_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_gaussian_noise_2_layer_call_and_return_conditional_losses_2356Ї
activation_2/PartitionedCallPartitionedCall1gaussian_noise_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_activation_2_layer_call_and_return_conditional_losses_2115┌
flatten_3/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         Z* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_flatten_3_layer_call_and_return_conditional_losses_2123ў
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0batch_normalization_3_2777batch_normalization_3_2779batch_normalization_3_2781batch_normalization_3_2783*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         Z*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1702С
dense/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0
dense_2786
dense_2788*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         %*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_2150┬
)dense/ActivityRegularizer/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *4
f/R-
+__inference_dense_activity_regularizer_1726u
dense/ActivityRegularizer/ShapeShape&dense/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:w
-dense/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/dense/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/dense/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╙
'dense/ActivityRegularizer/strided_sliceStridedSlice(dense/ActivityRegularizer/Shape:output:06dense/ActivityRegularizer/strided_slice/stack:output:08dense/ActivityRegularizer/strided_slice/stack_1:output:08dense/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskИ
dense/ActivityRegularizer/CastCast0dense/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: е
!dense/ActivityRegularizer/truedivRealDiv2dense/ActivityRegularizer/PartitionedCall:output:0"dense/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: с
activation_3/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         %* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_activation_3_layer_call_and_return_conditional_losses_2169Л
3CNNs_start/kernel/Regularizer/Square/ReadVariableOpReadVariableOpcnns_start_2721*&
_output_shapes
:		x*
dtype0Ь
$CNNs_start/kernel/Regularizer/SquareSquare;CNNs_start/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		x|
#CNNs_start/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             б
!CNNs_start/kernel/Regularizer/SumSum(CNNs_start/kernel/Regularizer/Square:y:0,CNNs_start/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#CNNs_start/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:г
!CNNs_start/kernel/Regularizer/mulMul,CNNs_start/kernel/Regularizer/mul/x:output:0*CNNs_start/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: Г
/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_2740*&
_output_shapes
:*
dtype0Ф
 conv2d/kernel/Regularizer/SquareSquare7conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:x
conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             Х
conv2d/kernel/Regularizer/SumSum$conv2d/kernel/Regularizer/Square:y:0(conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:Ч
conv2d/kernel/Regularizer/mulMul(conv2d/kernel/Regularizer/mul/x:output:0&conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: З
1conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_1_2758*&
_output_shapes
:*
dtype0Ш
"conv2d_1/kernel/Regularizer/SquareSquare9conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
!conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             Ы
conv2d_1/kernel/Regularizer/SumSum&conv2d_1/kernel/Regularizer/Square:y:0*conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:Э
conv2d_1/kernel/Regularizer/mulMul*conv2d_1/kernel/Regularizer/mul/x:output:0(conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: y
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp
dense_2786*
_output_shapes

:Z%*
dtype0К
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:Z%o
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Т
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:Ф
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: t
IdentityIdentity%activation_3/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         %e

Identity_1Identity%dense/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: ║
NoOpNoOp#^CNNs_start/StatefulPartitionedCall4^CNNs_start/kernel/Regularizer/Square/ReadVariableOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall^conv2d/StatefulPartitionedCall0^conv2d/kernel/Regularizer/Square/ReadVariableOp!^conv2d_1/StatefulPartitionedCall2^conv2d_1/kernel/Regularizer/Square/ReadVariableOp^dense/StatefulPartitionedCall/^dense/kernel/Regularizer/Square/ReadVariableOp'^gaussian_noise/StatefulPartitionedCall)^gaussian_noise_1/StatefulPartitionedCall)^gaussian_noise_2/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall1^photoreceptor_rods_reike/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*П
_input_shapes~
|:         ┤': : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
"CNNs_start/StatefulPartitionedCall"CNNs_start/StatefulPartitionedCall2j
3CNNs_start/kernel/Regularizer/Square/ReadVariableOp3CNNs_start/kernel/Regularizer/Square/ReadVariableOp2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2b
/conv2d/kernel/Regularizer/Square/ReadVariableOp/conv2d/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2f
1conv2d_1/kernel/Regularizer/Square/ReadVariableOp1conv2d_1/kernel/Regularizer/Square/ReadVariableOp2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2P
&gaussian_noise/StatefulPartitionedCall&gaussian_noise/StatefulPartitionedCall2T
(gaussian_noise_1/StatefulPartitionedCall(gaussian_noise_1/StatefulPartitionedCall2T
(gaussian_noise_2/StatefulPartitionedCall(gaussian_noise_2/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2d
0photoreceptor_rods_reike/StatefulPartitionedCall0photoreceptor_rods_reike/StatefulPartitionedCall:X T
0
_output_shapes
:         ┤'
 
_user_specified_nameinputs
ы
╥
.__inference_PRFR_CNN2D_RODS_layer_call_fn_3687

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20: 

unknown_21:x' 

unknown_22:x'$

unknown_23:		x

unknown_24:

unknown_25:	╨*

unknown_26:	╨*

unknown_27:	╨*

unknown_28:	╨*$

unknown_29:

unknown_30:

unknown_31:	╨

unknown_32:	╨

unknown_33:	╨

unknown_34:	╨$

unknown_35:

unknown_36:

unknown_37:Z

unknown_38:Z

unknown_39:Z

unknown_40:Z

unknown_41:Z

unknown_42:Z

unknown_43:Z

unknown_44:Z

unknown_45:Z%

unknown_46:%
identityИвStatefulPartitionedCall╫
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46*<
Tin5
321*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         %: *J
_read_only_resource_inputs,
*(	
 #$%&)*-./0*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_PRFR_CNN2D_RODS_layer_call_and_return_conditional_losses_2826o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         %`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*П
_input_shapes~
|:         ┤': : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         ┤'
 
_user_specified_nameinputs
║
c
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_4655

inputs
identityЮ
MaxPoolMaxPoolinputs*/
_output_shapes
:         *
data_formatNCHW*
ksize
*
paddingVALID*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
Т
f
J__inference_gaussian_noise_2_layer_call_and_return_conditional_losses_5021

inputs
identityV
IdentityIdentityinputs*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
ъ
b
F__inference_activation_1_layer_call_and_return_conditional_losses_4866

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:         	b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:         	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         	:W S
/
_output_shapes
:         	
 
_user_specified_nameinputs
жо
о
I__inference_PRFR_CNN2D_RODS_layer_call_and_return_conditional_losses_3192
input_1/
photoreceptor_rods_reike_3032:/
photoreceptor_rods_reike_3034:/
photoreceptor_rods_reike_3036:/
photoreceptor_rods_reike_3038:/
photoreceptor_rods_reike_3040:/
photoreceptor_rods_reike_3042:/
photoreceptor_rods_reike_3044:/
photoreceptor_rods_reike_3046:/
photoreceptor_rods_reike_3048:/
photoreceptor_rods_reike_3050:/
photoreceptor_rods_reike_3052:/
photoreceptor_rods_reike_3054:/
photoreceptor_rods_reike_3056:/
photoreceptor_rods_reike_3058:/
photoreceptor_rods_reike_3060:/
photoreceptor_rods_reike_3062:/
photoreceptor_rods_reike_3064:/
photoreceptor_rods_reike_3066:/
photoreceptor_rods_reike_3068:/
photoreceptor_rods_reike_3070:/
photoreceptor_rods_reike_3072:/
photoreceptor_rods_reike_3074:.
layer_normalization_3082:x'.
layer_normalization_3084:x')
cnns_start_3087:		x
cnns_start_3089:'
batch_normalization_3093:	╨*'
batch_normalization_3095:	╨*'
batch_normalization_3097:	╨*'
batch_normalization_3099:	╨*%
conv2d_3106:
conv2d_3108:)
batch_normalization_1_3112:	╨)
batch_normalization_1_3114:	╨)
batch_normalization_1_3116:	╨)
batch_normalization_1_3118:	╨'
conv2d_1_3124:
conv2d_1_3126:(
batch_normalization_2_3130:Z(
batch_normalization_2_3132:Z(
batch_normalization_2_3134:Z(
batch_normalization_2_3136:Z(
batch_normalization_3_3143:Z(
batch_normalization_3_3145:Z(
batch_normalization_3_3147:Z(
batch_normalization_3_3149:Z

dense_3152:Z%

dense_3154:%
identity

identity_1Ив"CNNs_start/StatefulPartitionedCallв3CNNs_start/kernel/Regularizer/Square/ReadVariableOpв+batch_normalization/StatefulPartitionedCallв-batch_normalization_1/StatefulPartitionedCallв-batch_normalization_2/StatefulPartitionedCallв-batch_normalization_3/StatefulPartitionedCallвconv2d/StatefulPartitionedCallв/conv2d/kernel/Regularizer/Square/ReadVariableOpв conv2d_1/StatefulPartitionedCallв1conv2d_1/kernel/Regularizer/Square/ReadVariableOpвdense/StatefulPartitionedCallв.dense/kernel/Regularizer/Square/ReadVariableOpв+layer_normalization/StatefulPartitionedCallв0photoreceptor_rods_reike/StatefulPartitionedCall╛
reshape/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         ┤Т	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_reshape_layer_call_and_return_conditional_losses_1746с
0photoreceptor_rods_reike/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0photoreceptor_rods_reike_3032photoreceptor_rods_reike_3034photoreceptor_rods_reike_3036photoreceptor_rods_reike_3038photoreceptor_rods_reike_3040photoreceptor_rods_reike_3042photoreceptor_rods_reike_3044photoreceptor_rods_reike_3046photoreceptor_rods_reike_3048photoreceptor_rods_reike_3050photoreceptor_rods_reike_3052photoreceptor_rods_reike_3054photoreceptor_rods_reike_3056photoreceptor_rods_reike_3058photoreceptor_rods_reike_3060photoreceptor_rods_reike_3062photoreceptor_rods_reike_3064photoreceptor_rods_reike_3066photoreceptor_rods_reike_3068photoreceptor_rods_reike_3070photoreceptor_rods_reike_3072photoreceptor_rods_reike_3074*"
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         ┤Т	*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_photoreceptor_rods_reike_layer_call_and_return_conditional_losses_1809ў
reshape_1/PartitionedCallPartitionedCall9photoreceptor_rods_reike/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ┤'* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_reshape_1_layer_call_and_return_conditional_losses_1869Е
,tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"    <           З
.tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                З
.tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            ь
&tf.__operators__.getitem/strided_sliceStridedSlice"reshape_1/PartitionedCall:output:05tf.__operators__.getitem/strided_slice/stack:output:07tf.__operators__.getitem/strided_slice/stack_1:output:07tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         x'*

begin_mask*
end_mask╩
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCall/tf.__operators__.getitem/strided_slice:output:0layer_normalization_3082layer_normalization_3084*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         x'*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_layer_normalization_layer_call_and_return_conditional_losses_1901л
"CNNs_start/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0cnns_start_3087cnns_start_3089*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_CNNs_start_layer_call_and_return_conditional_losses_1923▌
flatten/PartitionedCallPartitionedCall+CNNs_start/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ╨** 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_1935ь
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0batch_normalization_3093batch_normalization_3095batch_normalization_3097batch_normalization_3099*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ╨**&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_1397ё
reshape_2/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_reshape_2_layer_call_and_return_conditional_losses_1960ч
max_pooling2d/PartitionedCallPartitionedCall"reshape_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_1966э
gaussian_noise/PartitionedCallPartitionedCall&max_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_gaussian_noise_layer_call_and_return_conditional_losses_1972ц
activation/PartitionedCallPartitionedCall'gaussian_noise/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_activation_layer_call_and_return_conditional_losses_1979К
conv2d/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0conv2d_3106conv2d_3108*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_1997▌
flatten_1/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ╨* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_flatten_1_layer_call_and_return_conditional_losses_2009·
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0batch_normalization_1_3112batch_normalization_1_3114batch_normalization_1_3116batch_normalization_1_3118*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ╨*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1491є
reshape_3/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_reshape_3_layer_call_and_return_conditional_losses_2034э
 gaussian_noise_1/PartitionedCallPartitionedCall"reshape_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_gaussian_noise_1_layer_call_and_return_conditional_losses_2040ь
activation_1/PartitionedCallPartitionedCall)gaussian_noise_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_activation_1_layer_call_and_return_conditional_losses_2047Ф
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0conv2d_1_3124conv2d_1_3126*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_2065▐
flatten_2/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         Z* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_flatten_2_layer_call_and_return_conditional_losses_2077∙
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0batch_normalization_2_3130batch_normalization_2_3132batch_normalization_2_3134batch_normalization_2_3136*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         Z*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1573є
reshape_4/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_reshape_4_layer_call_and_return_conditional_losses_2102э
 gaussian_noise_2/PartitionedCallPartitionedCall"reshape_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_gaussian_noise_2_layer_call_and_return_conditional_losses_2108ь
activation_2/PartitionedCallPartitionedCall)gaussian_noise_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_activation_2_layer_call_and_return_conditional_losses_2115┌
flatten_3/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         Z* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_flatten_3_layer_call_and_return_conditional_losses_2123∙
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0batch_normalization_3_3143batch_normalization_3_3145batch_normalization_3_3147batch_normalization_3_3149*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         Z*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1655С
dense/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0
dense_3152
dense_3154*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         %*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_2150┬
)dense/ActivityRegularizer/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *4
f/R-
+__inference_dense_activity_regularizer_1726u
dense/ActivityRegularizer/ShapeShape&dense/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:w
-dense/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/dense/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/dense/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╙
'dense/ActivityRegularizer/strided_sliceStridedSlice(dense/ActivityRegularizer/Shape:output:06dense/ActivityRegularizer/strided_slice/stack:output:08dense/ActivityRegularizer/strided_slice/stack_1:output:08dense/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskИ
dense/ActivityRegularizer/CastCast0dense/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: е
!dense/ActivityRegularizer/truedivRealDiv2dense/ActivityRegularizer/PartitionedCall:output:0"dense/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: с
activation_3/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         %* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_activation_3_layer_call_and_return_conditional_losses_2169Л
3CNNs_start/kernel/Regularizer/Square/ReadVariableOpReadVariableOpcnns_start_3087*&
_output_shapes
:		x*
dtype0Ь
$CNNs_start/kernel/Regularizer/SquareSquare;CNNs_start/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		x|
#CNNs_start/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             б
!CNNs_start/kernel/Regularizer/SumSum(CNNs_start/kernel/Regularizer/Square:y:0,CNNs_start/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#CNNs_start/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:г
!CNNs_start/kernel/Regularizer/mulMul,CNNs_start/kernel/Regularizer/mul/x:output:0*CNNs_start/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: Г
/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_3106*&
_output_shapes
:*
dtype0Ф
 conv2d/kernel/Regularizer/SquareSquare7conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:x
conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             Х
conv2d/kernel/Regularizer/SumSum$conv2d/kernel/Regularizer/Square:y:0(conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:Ч
conv2d/kernel/Regularizer/mulMul(conv2d/kernel/Regularizer/mul/x:output:0&conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: З
1conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_1_3124*&
_output_shapes
:*
dtype0Ш
"conv2d_1/kernel/Regularizer/SquareSquare9conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
!conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             Ы
conv2d_1/kernel/Regularizer/SumSum&conv2d_1/kernel/Regularizer/Square:y:0*conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:Э
conv2d_1/kernel/Regularizer/mulMul*conv2d_1/kernel/Regularizer/mul/x:output:0(conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: y
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp
dense_3152*
_output_shapes

:Z%*
dtype0К
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:Z%o
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Т
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:Ф
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: t
IdentityIdentity%activation_3/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         %e

Identity_1Identity%dense/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: ╗
NoOpNoOp#^CNNs_start/StatefulPartitionedCall4^CNNs_start/kernel/Regularizer/Square/ReadVariableOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall^conv2d/StatefulPartitionedCall0^conv2d/kernel/Regularizer/Square/ReadVariableOp!^conv2d_1/StatefulPartitionedCall2^conv2d_1/kernel/Regularizer/Square/ReadVariableOp^dense/StatefulPartitionedCall/^dense/kernel/Regularizer/Square/ReadVariableOp,^layer_normalization/StatefulPartitionedCall1^photoreceptor_rods_reike/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*П
_input_shapes~
|:         ┤': : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
"CNNs_start/StatefulPartitionedCall"CNNs_start/StatefulPartitionedCall2j
3CNNs_start/kernel/Regularizer/Square/ReadVariableOp3CNNs_start/kernel/Regularizer/Square/ReadVariableOp2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2b
/conv2d/kernel/Regularizer/Square/ReadVariableOp/conv2d/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2f
1conv2d_1/kernel/Regularizer/Square/ReadVariableOp1conv2d_1/kernel/Regularizer/Square/ReadVariableOp2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2d
0photoreceptor_rods_reike/StatefulPartitionedCall0photoreceptor_rods_reike/StatefulPartitionedCall:Y U
0
_output_shapes
:         ┤'
!
_user_specified_name	input_1
п
╙
4__inference_batch_normalization_1_layer_call_fn_4745

inputs
unknown:	╨
	unknown_0:	╨
	unknown_1:	╨
	unknown_2:	╨
identityИвStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ╨*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1491p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ╨`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ╨: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ╨
 
_user_specified_nameinputs
│
H
,__inference_max_pooling2d_layer_call_fn_4640

inputs
identity╪
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_1464Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs"ВL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*╕
serving_defaultд
D
input_19
serving_default_input_1:0         ┤'@
activation_30
StatefulPartitionedCall:0         %tensorflow/serving/predict:┘╩
Ў
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer-4
layer_with_weights-1
layer-5
layer_with_weights-2
layer-6
layer-7
	layer_with_weights-3
	layer-8

layer-9
layer-10
layer-11
layer-12
layer_with_weights-4
layer-13
layer-14
layer_with_weights-5
layer-15
layer-16
layer-17
layer-18
layer_with_weights-6
layer-19
layer-20
layer_with_weights-7
layer-21
layer-22
layer-23
layer-24
layer-25
layer_with_weights-8
layer-26
layer_with_weights-9
layer-27
layer-28
	optimizer
	variables
 trainable_variables
!regularization_losses
"	keras_api
#
signatures
щ__call__
+ъ&call_and_return_all_conditional_losses
ы_default_save_signature"
_tf_keras_network
"
_tf_keras_input_layer
з
$	variables
%trainable_variables
&regularization_losses
'	keras_api
ь__call__
+э&call_and_return_all_conditional_losses"
_tf_keras_layer
М
	(sigma
)sigma_scaleFac
*phi
+phi_scaleFac
,eta
-eta_scaleFac
.beta
/beta_scaleFac
0cgmp2cur
1cgmphill
2cgmphill_scaleFac
	3cdark
4betaSlow
5betaSlow_scaleFac
6hillcoef
7hillcoef_scaleFac
8hillaffinity
9hillaffinity_scaleFac
	:gamma
;gamma_scaleFac
	<gdark
=gdark_scaleFac
>	variables
?trainable_variables
@regularization_losses
A	keras_api
ю__call__
+я&call_and_return_all_conditional_losses"
_tf_keras_layer
з
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
Ё__call__
+ё&call_and_return_all_conditional_losses"
_tf_keras_layer
(
F	keras_api"
_tf_keras_layer
╞
Gaxis
	Hgamma
Ibeta
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
Є__call__
+є&call_and_return_all_conditional_losses"
_tf_keras_layer
╜

Nkernel
Obias
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
Ї__call__
+ї&call_and_return_all_conditional_losses"
_tf_keras_layer
з
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
Ў__call__
+ў&call_and_return_all_conditional_losses"
_tf_keras_layer
ь
Xaxis
	Ygamma
Zbeta
[moving_mean
\moving_variance
]	variables
^trainable_variables
_regularization_losses
`	keras_api
°__call__
+∙&call_and_return_all_conditional_losses"
_tf_keras_layer
з
a	variables
btrainable_variables
cregularization_losses
d	keras_api
·__call__
+√&call_and_return_all_conditional_losses"
_tf_keras_layer
з
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
№__call__
+¤&call_and_return_all_conditional_losses"
_tf_keras_layer
з
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
■__call__
+ &call_and_return_all_conditional_losses"
_tf_keras_layer
з
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
А__call__
+Б&call_and_return_all_conditional_losses"
_tf_keras_layer
╜

qkernel
rbias
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
В__call__
+Г&call_and_return_all_conditional_losses"
_tf_keras_layer
з
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
Д__call__
+Е&call_and_return_all_conditional_losses"
_tf_keras_layer
Ё
{axis
	|gamma
}beta
~moving_mean
moving_variance
А	variables
Бtrainable_variables
Вregularization_losses
Г	keras_api
Ж__call__
+З&call_and_return_all_conditional_losses"
_tf_keras_layer
л
Д	variables
Еtrainable_variables
Жregularization_losses
З	keras_api
И__call__
+Й&call_and_return_all_conditional_losses"
_tf_keras_layer
л
И	variables
Йtrainable_variables
Кregularization_losses
Л	keras_api
К__call__
+Л&call_and_return_all_conditional_losses"
_tf_keras_layer
л
М	variables
Нtrainable_variables
Оregularization_losses
П	keras_api
М__call__
+Н&call_and_return_all_conditional_losses"
_tf_keras_layer
├
Рkernel
	Сbias
Т	variables
Уtrainable_variables
Фregularization_losses
Х	keras_api
О__call__
+П&call_and_return_all_conditional_losses"
_tf_keras_layer
л
Ц	variables
Чtrainable_variables
Шregularization_losses
Щ	keras_api
Р__call__
+С&call_and_return_all_conditional_losses"
_tf_keras_layer
ї
	Ъaxis

Ыgamma
	Ьbeta
Эmoving_mean
Юmoving_variance
Я	variables
аtrainable_variables
бregularization_losses
в	keras_api
Т__call__
+У&call_and_return_all_conditional_losses"
_tf_keras_layer
л
г	variables
дtrainable_variables
еregularization_losses
ж	keras_api
Ф__call__
+Х&call_and_return_all_conditional_losses"
_tf_keras_layer
л
з	variables
иtrainable_variables
йregularization_losses
к	keras_api
Ц__call__
+Ч&call_and_return_all_conditional_losses"
_tf_keras_layer
л
л	variables
мtrainable_variables
нregularization_losses
о	keras_api
Ш__call__
+Щ&call_and_return_all_conditional_losses"
_tf_keras_layer
л
п	variables
░trainable_variables
▒regularization_losses
▓	keras_api
Ъ__call__
+Ы&call_and_return_all_conditional_losses"
_tf_keras_layer
ї
	│axis

┤gamma
	╡beta
╢moving_mean
╖moving_variance
╕	variables
╣trainable_variables
║regularization_losses
╗	keras_api
Ь__call__
+Э&call_and_return_all_conditional_losses"
_tf_keras_layer
├
╝kernel
	╜bias
╛	variables
┐trainable_variables
└regularization_losses
┴	keras_api
Ю__call__
+Я&call_and_return_all_conditional_losses"
_tf_keras_layer
л
┬	variables
├trainable_variables
─regularization_losses
┼	keras_api
а__call__
+б&call_and_return_all_conditional_losses"
_tf_keras_layer
"
	optimizer
в
(0
*1
,2
.3
)4
+5
-6
/7
08
19
210
311
412
513
614
715
816
917
:18
;19
<20
=21
H22
I23
N24
O25
Y26
Z27
[28
\29
q30
r31
|32
}33
~34
35
Р36
С37
Ы38
Ь39
Э40
Ю41
┤42
╡43
╢44
╖45
╝46
╜47"
trackable_list_wrapper
╬
(0
*1
,2
.3
H4
I5
N6
O7
Y8
Z9
q10
r11
|12
}13
Р14
С15
Ы16
Ь17
┤18
╡19
╝20
╜21"
trackable_list_wrapper
@
в0
г1
д2
е3"
trackable_list_wrapper
╙
╞non_trainable_variables
╟layers
╚metrics
 ╔layer_regularization_losses
╩layer_metrics
	variables
 trainable_variables
!regularization_losses
щ__call__
ы_default_save_signature
+ъ&call_and_return_all_conditional_losses
'ъ"call_and_return_conditional_losses"
_generic_user_object
-
жserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
╦non_trainable_variables
╠layers
═metrics
 ╬layer_regularization_losses
╧layer_metrics
$	variables
%trainable_variables
&regularization_losses
ь__call__
+э&call_and_return_all_conditional_losses
'э"call_and_return_conditional_losses"
_generic_user_object
0:.2photoreceptor_rods_reike/sigma
7:52'photoreceptor_rods_reike/sigma_scaleFac
.:,2photoreceptor_rods_reike/phi
5:32%photoreceptor_rods_reike/phi_scaleFac
.:,2photoreceptor_rods_reike/eta
5:32%photoreceptor_rods_reike/eta_scaleFac
/:-2photoreceptor_rods_reike/beta
6:42&photoreceptor_rods_reike/beta_scaleFac
1:/2!photoreceptor_rods_reike/cgmp2cur
1:/2!photoreceptor_rods_reike/cgmphill
::82*photoreceptor_rods_reike/cgmphill_scaleFac
.:,2photoreceptor_rods_reike/cdark
1:/2!photoreceptor_rods_reike/betaSlow
::82*photoreceptor_rods_reike/betaSlow_scaleFac
1:/2!photoreceptor_rods_reike/hillcoef
::82*photoreceptor_rods_reike/hillcoef_scaleFac
5:32%photoreceptor_rods_reike/hillaffinity
>:<2.photoreceptor_rods_reike/hillaffinity_scaleFac
.:,2photoreceptor_rods_reike/gamma
7:52'photoreceptor_rods_reike/gamma_scaleFac
.:,2photoreceptor_rods_reike/gdark
7:52'photoreceptor_rods_reike/gdark_scaleFac
╞
(0
*1
,2
.3
)4
+5
-6
/7
08
19
210
311
412
513
614
715
816
917
:18
;19
<20
=21"
trackable_list_wrapper
<
(0
*1
,2
.3"
trackable_list_wrapper
 "
trackable_list_wrapper
╡
╨non_trainable_variables
╤layers
╥metrics
 ╙layer_regularization_losses
╘layer_metrics
>	variables
?trainable_variables
@regularization_losses
ю__call__
+я&call_and_return_all_conditional_losses
'я"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
╒non_trainable_variables
╓layers
╫metrics
 ╪layer_regularization_losses
┘layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
Ё__call__
+ё&call_and_return_all_conditional_losses
'ё"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
 "
trackable_list_wrapper
/:-x'2layer_normalization/gamma
.:,x'2layer_normalization/beta
.
H0
I1"
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
 "
trackable_list_wrapper
╡
┌non_trainable_variables
█layers
▄metrics
 ▌layer_regularization_losses
▐layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
Є__call__
+є&call_and_return_all_conditional_losses
'є"call_and_return_conditional_losses"
_generic_user_object
+:)		x2CNNs_start/kernel
:2CNNs_start/bias
.
N0
O1"
trackable_list_wrapper
.
N0
O1"
trackable_list_wrapper
(
в0"
trackable_list_wrapper
╡
▀non_trainable_variables
рlayers
сmetrics
 тlayer_regularization_losses
уlayer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
Ї__call__
+ї&call_and_return_all_conditional_losses
'ї"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
фnon_trainable_variables
хlayers
цmetrics
 чlayer_regularization_losses
шlayer_metrics
T	variables
Utrainable_variables
Vregularization_losses
Ў__call__
+ў&call_and_return_all_conditional_losses
'ў"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
(:&╨*2batch_normalization/gamma
':%╨*2batch_normalization/beta
0:.╨* (2batch_normalization/moving_mean
4:2╨* (2#batch_normalization/moving_variance
<
Y0
Z1
[2
\3"
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
 "
trackable_list_wrapper
╡
щnon_trainable_variables
ъlayers
ыmetrics
 ьlayer_regularization_losses
эlayer_metrics
]	variables
^trainable_variables
_regularization_losses
°__call__
+∙&call_and_return_all_conditional_losses
'∙"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
юnon_trainable_variables
яlayers
Ёmetrics
 ёlayer_regularization_losses
Єlayer_metrics
a	variables
btrainable_variables
cregularization_losses
·__call__
+√&call_and_return_all_conditional_losses
'√"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
єnon_trainable_variables
Їlayers
їmetrics
 Ўlayer_regularization_losses
ўlayer_metrics
e	variables
ftrainable_variables
gregularization_losses
№__call__
+¤&call_and_return_all_conditional_losses
'¤"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
°non_trainable_variables
∙layers
·metrics
 √layer_regularization_losses
№layer_metrics
i	variables
jtrainable_variables
kregularization_losses
■__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
¤non_trainable_variables
■layers
 metrics
 Аlayer_regularization_losses
Бlayer_metrics
m	variables
ntrainable_variables
oregularization_losses
А__call__
+Б&call_and_return_all_conditional_losses
'Б"call_and_return_conditional_losses"
_generic_user_object
':%2conv2d/kernel
:2conv2d/bias
.
q0
r1"
trackable_list_wrapper
.
q0
r1"
trackable_list_wrapper
(
г0"
trackable_list_wrapper
╡
Вnon_trainable_variables
Гlayers
Дmetrics
 Еlayer_regularization_losses
Жlayer_metrics
s	variables
ttrainable_variables
uregularization_losses
В__call__
+Г&call_and_return_all_conditional_losses
'Г"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
Зnon_trainable_variables
Иlayers
Йmetrics
 Кlayer_regularization_losses
Лlayer_metrics
w	variables
xtrainable_variables
yregularization_losses
Д__call__
+Е&call_and_return_all_conditional_losses
'Е"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(╨2batch_normalization_1/gamma
):'╨2batch_normalization_1/beta
2:0╨ (2!batch_normalization_1/moving_mean
6:4╨ (2%batch_normalization_1/moving_variance
<
|0
}1
~2
3"
trackable_list_wrapper
.
|0
}1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Мnon_trainable_variables
Нlayers
Оmetrics
 Пlayer_regularization_losses
Рlayer_metrics
А	variables
Бtrainable_variables
Вregularization_losses
Ж__call__
+З&call_and_return_all_conditional_losses
'З"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Сnon_trainable_variables
Тlayers
Уmetrics
 Фlayer_regularization_losses
Хlayer_metrics
Д	variables
Еtrainable_variables
Жregularization_losses
И__call__
+Й&call_and_return_all_conditional_losses
'Й"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Цnon_trainable_variables
Чlayers
Шmetrics
 Щlayer_regularization_losses
Ъlayer_metrics
И	variables
Йtrainable_variables
Кregularization_losses
К__call__
+Л&call_and_return_all_conditional_losses
'Л"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Ыnon_trainable_variables
Ьlayers
Эmetrics
 Юlayer_regularization_losses
Яlayer_metrics
М	variables
Нtrainable_variables
Оregularization_losses
М__call__
+Н&call_and_return_all_conditional_losses
'Н"call_and_return_conditional_losses"
_generic_user_object
):'2conv2d_1/kernel
:2conv2d_1/bias
0
Р0
С1"
trackable_list_wrapper
0
Р0
С1"
trackable_list_wrapper
(
д0"
trackable_list_wrapper
╕
аnon_trainable_variables
бlayers
вmetrics
 гlayer_regularization_losses
дlayer_metrics
Т	variables
Уtrainable_variables
Фregularization_losses
О__call__
+П&call_and_return_all_conditional_losses
'П"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
еnon_trainable_variables
жlayers
зmetrics
 иlayer_regularization_losses
йlayer_metrics
Ц	variables
Чtrainable_variables
Шregularization_losses
Р__call__
+С&call_and_return_all_conditional_losses
'С"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'Z2batch_normalization_2/gamma
(:&Z2batch_normalization_2/beta
1:/Z (2!batch_normalization_2/moving_mean
5:3Z (2%batch_normalization_2/moving_variance
@
Ы0
Ь1
Э2
Ю3"
trackable_list_wrapper
0
Ы0
Ь1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
кnon_trainable_variables
лlayers
мmetrics
 нlayer_regularization_losses
оlayer_metrics
Я	variables
аtrainable_variables
бregularization_losses
Т__call__
+У&call_and_return_all_conditional_losses
'У"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
пnon_trainable_variables
░layers
▒metrics
 ▓layer_regularization_losses
│layer_metrics
г	variables
дtrainable_variables
еregularization_losses
Ф__call__
+Х&call_and_return_all_conditional_losses
'Х"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
┤non_trainable_variables
╡layers
╢metrics
 ╖layer_regularization_losses
╕layer_metrics
з	variables
иtrainable_variables
йregularization_losses
Ц__call__
+Ч&call_and_return_all_conditional_losses
'Ч"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
╣non_trainable_variables
║layers
╗metrics
 ╝layer_regularization_losses
╜layer_metrics
л	variables
мtrainable_variables
нregularization_losses
Ш__call__
+Щ&call_and_return_all_conditional_losses
'Щ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
╛non_trainable_variables
┐layers
└metrics
 ┴layer_regularization_losses
┬layer_metrics
п	variables
░trainable_variables
▒regularization_losses
Ъ__call__
+Ы&call_and_return_all_conditional_losses
'Ы"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'Z2batch_normalization_3/gamma
(:&Z2batch_normalization_3/beta
1:/Z (2!batch_normalization_3/moving_mean
5:3Z (2%batch_normalization_3/moving_variance
@
┤0
╡1
╢2
╖3"
trackable_list_wrapper
0
┤0
╡1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
├non_trainable_variables
─layers
┼metrics
 ╞layer_regularization_losses
╟layer_metrics
╕	variables
╣trainable_variables
║regularization_losses
Ь__call__
+Э&call_and_return_all_conditional_losses
'Э"call_and_return_conditional_losses"
_generic_user_object
:Z%2dense/kernel
:%2
dense/bias
0
╝0
╜1"
trackable_list_wrapper
0
╝0
╜1"
trackable_list_wrapper
(
е0"
trackable_list_wrapper
╓
╚non_trainable_variables
╔layers
╩metrics
 ╦layer_regularization_losses
╠layer_metrics
╛	variables
┐trainable_variables
└regularization_losses
Ю__call__
зactivity_regularizer_fn
+Я&call_and_return_all_conditional_losses
'и"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
═non_trainable_variables
╬layers
╧metrics
 ╨layer_regularization_losses
╤layer_metrics
┬	variables
├trainable_variables
─regularization_losses
а__call__
+б&call_and_return_all_conditional_losses
'б"call_and_return_conditional_losses"
_generic_user_object
ъ
)0
+1
-2
/3
04
15
26
37
48
59
610
711
812
913
:14
;15
<16
=17
[18
\19
~20
21
Э22
Ю23
╢24
╖25"
trackable_list_wrapper
■
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28"
trackable_list_wrapper
@
╥0
╙1
╘2
╒3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ж
)0
+1
-2
/3
04
15
26
37
48
59
610
711
812
913
:14
;15
<16
=17"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
в0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
[0
\1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
г0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
~0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
д0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
Э0
Ю1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
╢0
╖1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
е0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
R

╓total

╫count
╪	variables
┘	keras_api"
_tf_keras_metric
c

┌total

█count
▄
_fn_kwargs
▌	variables
▐	keras_api"
_tf_keras_metric
c

▀total

рcount
с
_fn_kwargs
т	variables
у	keras_api"
_tf_keras_metric
c

фtotal

хcount
ц
_fn_kwargs
ч	variables
ш	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
╓0
╫1"
trackable_list_wrapper
.
╪	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
┌0
█1"
trackable_list_wrapper
.
▌	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
▀0
р1"
trackable_list_wrapper
.
т	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
ф0
х1"
trackable_list_wrapper
.
ч	variables"
_generic_user_object
Ж2Г
.__inference_PRFR_CNN2D_RODS_layer_call_fn_2297
.__inference_PRFR_CNN2D_RODS_layer_call_fn_3585
.__inference_PRFR_CNN2D_RODS_layer_call_fn_3687
.__inference_PRFR_CNN2D_RODS_layer_call_fn_3028└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Є2я
I__inference_PRFR_CNN2D_RODS_layer_call_and_return_conditional_losses_3961
I__inference_PRFR_CNN2D_RODS_layer_call_and_return_conditional_losses_4312
I__inference_PRFR_CNN2D_RODS_layer_call_and_return_conditional_losses_3192
I__inference_PRFR_CNN2D_RODS_layer_call_and_return_conditional_losses_3356└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╩B╟
__inference__wrapped_model_1373input_1"Ш
С▓Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╨2═
&__inference_reshape_layer_call_fn_4317в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ы2ш
A__inference_reshape_layer_call_and_return_conditional_losses_4330в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
с2▐
7__inference_photoreceptor_rods_reike_layer_call_fn_4379в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
№2∙
R__inference_photoreceptor_rods_reike_layer_call_and_return_conditional_losses_4440в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╥2╧
(__inference_reshape_1_layer_call_fn_4445в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
э2ъ
C__inference_reshape_1_layer_call_and_return_conditional_losses_4459в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
▄2┘
2__inference_layer_normalization_layer_call_fn_4468в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ў2Ї
M__inference_layer_normalization_layer_call_and_return_conditional_losses_4494в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╙2╨
)__inference_CNNs_start_layer_call_fn_4509в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ю2ы
D__inference_CNNs_start_layer_call_and_return_conditional_losses_4525в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╨2═
&__inference_flatten_layer_call_fn_4530в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ы2ш
A__inference_flatten_layer_call_and_return_conditional_losses_4536в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
в2Я
2__inference_batch_normalization_layer_call_fn_4549
2__inference_batch_normalization_layer_call_fn_4562┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╪2╒
M__inference_batch_normalization_layer_call_and_return_conditional_losses_4582
M__inference_batch_normalization_layer_call_and_return_conditional_losses_4616┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╥2╧
(__inference_reshape_2_layer_call_fn_4621в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
э2ъ
C__inference_reshape_2_layer_call_and_return_conditional_losses_4635в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Д2Б
,__inference_max_pooling2d_layer_call_fn_4640
,__inference_max_pooling2d_layer_call_fn_4645в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
║2╖
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_4650
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_4655в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ш2Х
-__inference_gaussian_noise_layer_call_fn_4660
-__inference_gaussian_noise_layer_call_fn_4665┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╬2╦
H__inference_gaussian_noise_layer_call_and_return_conditional_losses_4669
H__inference_gaussian_noise_layer_call_and_return_conditional_losses_4680┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╙2╨
)__inference_activation_layer_call_fn_4685в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ю2ы
D__inference_activation_layer_call_and_return_conditional_losses_4690в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╧2╠
%__inference_conv2d_layer_call_fn_4705в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ъ2ч
@__inference_conv2d_layer_call_and_return_conditional_losses_4721в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╥2╧
(__inference_flatten_1_layer_call_fn_4726в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
э2ъ
C__inference_flatten_1_layer_call_and_return_conditional_losses_4732в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ж2г
4__inference_batch_normalization_1_layer_call_fn_4745
4__inference_batch_normalization_1_layer_call_fn_4758┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
▄2┘
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4778
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4812┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╥2╧
(__inference_reshape_3_layer_call_fn_4817в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
э2ъ
C__inference_reshape_3_layer_call_and_return_conditional_losses_4831в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ь2Щ
/__inference_gaussian_noise_1_layer_call_fn_4836
/__inference_gaussian_noise_1_layer_call_fn_4841┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╥2╧
J__inference_gaussian_noise_1_layer_call_and_return_conditional_losses_4845
J__inference_gaussian_noise_1_layer_call_and_return_conditional_losses_4856┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╒2╥
+__inference_activation_1_layer_call_fn_4861в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ё2э
F__inference_activation_1_layer_call_and_return_conditional_losses_4866в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╤2╬
'__inference_conv2d_1_layer_call_fn_4881в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ь2щ
B__inference_conv2d_1_layer_call_and_return_conditional_losses_4897в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╥2╧
(__inference_flatten_2_layer_call_fn_4902в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
э2ъ
C__inference_flatten_2_layer_call_and_return_conditional_losses_4908в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ж2г
4__inference_batch_normalization_2_layer_call_fn_4921
4__inference_batch_normalization_2_layer_call_fn_4934┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
▄2┘
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_4954
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_4988┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╥2╧
(__inference_reshape_4_layer_call_fn_4993в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
э2ъ
C__inference_reshape_4_layer_call_and_return_conditional_losses_5007в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ь2Щ
/__inference_gaussian_noise_2_layer_call_fn_5012
/__inference_gaussian_noise_2_layer_call_fn_5017┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╥2╧
J__inference_gaussian_noise_2_layer_call_and_return_conditional_losses_5021
J__inference_gaussian_noise_2_layer_call_and_return_conditional_losses_5032┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╒2╥
+__inference_activation_2_layer_call_fn_5037в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ё2э
F__inference_activation_2_layer_call_and_return_conditional_losses_5042в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╥2╧
(__inference_flatten_3_layer_call_fn_5047в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
э2ъ
C__inference_flatten_3_layer_call_and_return_conditional_losses_5053в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ж2г
4__inference_batch_normalization_3_layer_call_fn_5066
4__inference_batch_normalization_3_layer_call_fn_5079┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
▄2┘
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_5099
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_5133┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╬2╦
$__inference_dense_layer_call_fn_5148в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
э2ъ
C__inference_dense_layer_call_and_return_all_conditional_losses_5159в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╒2╥
+__inference_activation_3_layer_call_fn_5164в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ё2э
F__inference_activation_3_layer_call_and_return_conditional_losses_5169в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
▒2о
__inference_loss_fn_0_5180П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в 
▒2о
__inference_loss_fn_1_5191П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в 
▒2о
__inference_loss_fn_2_5202П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в 
▒2о
__inference_loss_fn_3_5213П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в 
╔B╞
"__inference_signature_wrapper_3483input_1"Ф
Н▓Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
▄2┘
+__inference_dense_activity_regularizer_1726й
Ф▓Р
FullArgSpec
argsЪ
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в
	К
щ2ц
?__inference_dense_layer_call_and_return_conditional_losses_5229в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 ┤
D__inference_CNNs_start_layer_call_and_return_conditional_losses_4525lNO7в4
-в*
(К%
inputs         x'
к "-в*
#К 
0         
Ъ М
)__inference_CNNs_start_layer_call_fn_4509_NO7в4
-в*
(К%
inputs         x'
к " К         Д
I__inference_PRFR_CNN2D_RODS_layer_call_and_return_conditional_losses_3192╢<()*+,-12./456789:;<=03HINO\Y[Zqr|~}РСЮЫЭЬ╖┤╢╡╝╜Aв>
7в4
*К'
input_1         ┤'
p 

 
к "3в0
К
0         %
Ъ
К	
1/0 Д
I__inference_PRFR_CNN2D_RODS_layer_call_and_return_conditional_losses_3356╢<()*+,-12./456789:;<=03HINO[\YZqr~|}РСЭЮЫЬ╢╖┤╡╝╜Aв>
7в4
*К'
input_1         ┤'
p

 
к "3в0
К
0         %
Ъ
К	
1/0 Г
I__inference_PRFR_CNN2D_RODS_layer_call_and_return_conditional_losses_3961╡<()*+,-12./456789:;<=03HINO\Y[Zqr|~}РСЮЫЭЬ╖┤╢╡╝╜@в=
6в3
)К&
inputs         ┤'
p 

 
к "3в0
К
0         %
Ъ
К	
1/0 Г
I__inference_PRFR_CNN2D_RODS_layer_call_and_return_conditional_losses_4312╡<()*+,-12./456789:;<=03HINO[\YZqr~|}РСЭЮЫЬ╢╖┤╡╝╜@в=
6в3
)К&
inputs         ┤'
p

 
к "3в0
К
0         %
Ъ
К	
1/0 ╬
.__inference_PRFR_CNN2D_RODS_layer_call_fn_2297Ы<()*+,-12./456789:;<=03HINO\Y[Zqr|~}РСЮЫЭЬ╖┤╢╡╝╜Aв>
7в4
*К'
input_1         ┤'
p 

 
к "К         %╬
.__inference_PRFR_CNN2D_RODS_layer_call_fn_3028Ы<()*+,-12./456789:;<=03HINO[\YZqr~|}РСЭЮЫЬ╢╖┤╡╝╜Aв>
7в4
*К'
input_1         ┤'
p

 
к "К         %═
.__inference_PRFR_CNN2D_RODS_layer_call_fn_3585Ъ<()*+,-12./456789:;<=03HINO\Y[Zqr|~}РСЮЫЭЬ╖┤╢╡╝╜@в=
6в3
)К&
inputs         ┤'
p 

 
к "К         %═
.__inference_PRFR_CNN2D_RODS_layer_call_fn_3687Ъ<()*+,-12./456789:;<=03HINO[\YZqr~|}РСЭЮЫЬ╢╖┤╡╝╜@в=
6в3
)К&
inputs         ┤'
p

 
к "К         %┌
__inference__wrapped_model_1373╢<()*+,-12./456789:;<=03HINO\Y[Zqr|~}РСЮЫЭЬ╖┤╢╡╝╜9в6
/в,
*К'
input_1         ┤'
к ";к8
6
activation_3&К#
activation_3         %▓
F__inference_activation_1_layer_call_and_return_conditional_losses_4866h7в4
-в*
(К%
inputs         	
к "-в*
#К 
0         	
Ъ К
+__inference_activation_1_layer_call_fn_4861[7в4
-в*
(К%
inputs         	
к " К         	▓
F__inference_activation_2_layer_call_and_return_conditional_losses_5042h7в4
-в*
(К%
inputs         
к "-в*
#К 
0         
Ъ К
+__inference_activation_2_layer_call_fn_5037[7в4
-в*
(К%
inputs         
к " К         в
F__inference_activation_3_layer_call_and_return_conditional_losses_5169X/в,
%в"
 К
inputs         %
к "%в"
К
0         %
Ъ z
+__inference_activation_3_layer_call_fn_5164K/в,
%в"
 К
inputs         %
к "К         %░
D__inference_activation_layer_call_and_return_conditional_losses_4690h7в4
-в*
(К%
inputs         
к "-в*
#К 
0         
Ъ И
)__inference_activation_layer_call_fn_4685[7в4
-в*
(К%
inputs         
к " К         ╖
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4778d|~}4в1
*в'
!К
inputs         ╨
p 
к "&в#
К
0         ╨
Ъ ╖
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4812d~|}4в1
*в'
!К
inputs         ╨
p
к "&в#
К
0         ╨
Ъ П
4__inference_batch_normalization_1_layer_call_fn_4745W|~}4в1
*в'
!К
inputs         ╨
p 
к "К         ╨П
4__inference_batch_normalization_1_layer_call_fn_4758W~|}4в1
*в'
!К
inputs         ╨
p
к "К         ╨╣
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_4954fЮЫЭЬ3в0
)в&
 К
inputs         Z
p 
к "%в"
К
0         Z
Ъ ╣
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_4988fЭЮЫЬ3в0
)в&
 К
inputs         Z
p
к "%в"
К
0         Z
Ъ С
4__inference_batch_normalization_2_layer_call_fn_4921YЮЫЭЬ3в0
)в&
 К
inputs         Z
p 
к "К         ZС
4__inference_batch_normalization_2_layer_call_fn_4934YЭЮЫЬ3в0
)в&
 К
inputs         Z
p
к "К         Z╣
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_5099f╖┤╢╡3в0
)в&
 К
inputs         Z
p 
к "%в"
К
0         Z
Ъ ╣
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_5133f╢╖┤╡3в0
)в&
 К
inputs         Z
p
к "%в"
К
0         Z
Ъ С
4__inference_batch_normalization_3_layer_call_fn_5066Y╖┤╢╡3в0
)в&
 К
inputs         Z
p 
к "К         ZС
4__inference_batch_normalization_3_layer_call_fn_5079Y╢╖┤╡3в0
)в&
 К
inputs         Z
p
к "К         Z╡
M__inference_batch_normalization_layer_call_and_return_conditional_losses_4582d\Y[Z4в1
*в'
!К
inputs         ╨*
p 
к "&в#
К
0         ╨*
Ъ ╡
M__inference_batch_normalization_layer_call_and_return_conditional_losses_4616d[\YZ4в1
*в'
!К
inputs         ╨*
p
к "&в#
К
0         ╨*
Ъ Н
2__inference_batch_normalization_layer_call_fn_4549W\Y[Z4в1
*в'
!К
inputs         ╨*
p 
к "К         ╨*Н
2__inference_batch_normalization_layer_call_fn_4562W[\YZ4в1
*в'
!К
inputs         ╨*
p
к "К         ╨*┤
B__inference_conv2d_1_layer_call_and_return_conditional_losses_4897nРС7в4
-в*
(К%
inputs         	
к "-в*
#К 
0         
Ъ М
'__inference_conv2d_1_layer_call_fn_4881aРС7в4
-в*
(К%
inputs         	
к " К         ░
@__inference_conv2d_layer_call_and_return_conditional_losses_4721lqr7в4
-в*
(К%
inputs         
к "-в*
#К 
0         	
Ъ И
%__inference_conv2d_layer_call_fn_4705_qr7в4
-в*
(К%
inputs         
к " К         	U
+__inference_dense_activity_regularizer_1726&в
в
К	
x
к "К │
C__inference_dense_layer_call_and_return_all_conditional_losses_5159l╝╜/в,
%в"
 К
inputs         Z
к "3в0
К
0         %
Ъ
К	
1/0 б
?__inference_dense_layer_call_and_return_conditional_losses_5229^╝╜/в,
%в"
 К
inputs         Z
к "%в"
К
0         %
Ъ y
$__inference_dense_layer_call_fn_5148Q╝╜/в,
%в"
 К
inputs         Z
к "К         %и
C__inference_flatten_1_layer_call_and_return_conditional_losses_4732a7в4
-в*
(К%
inputs         	
к "&в#
К
0         ╨
Ъ А
(__inference_flatten_1_layer_call_fn_4726T7в4
-в*
(К%
inputs         	
к "К         ╨з
C__inference_flatten_2_layer_call_and_return_conditional_losses_4908`7в4
-в*
(К%
inputs         
к "%в"
К
0         Z
Ъ 
(__inference_flatten_2_layer_call_fn_4902S7в4
-в*
(К%
inputs         
к "К         Zз
C__inference_flatten_3_layer_call_and_return_conditional_losses_5053`7в4
-в*
(К%
inputs         
к "%в"
К
0         Z
Ъ 
(__inference_flatten_3_layer_call_fn_5047S7в4
-в*
(К%
inputs         
к "К         Zж
A__inference_flatten_layer_call_and_return_conditional_losses_4536a7в4
-в*
(К%
inputs         
к "&в#
К
0         ╨*
Ъ ~
&__inference_flatten_layer_call_fn_4530T7в4
-в*
(К%
inputs         
к "К         ╨*║
J__inference_gaussian_noise_1_layer_call_and_return_conditional_losses_4845l;в8
1в.
(К%
inputs         	
p 
к "-в*
#К 
0         	
Ъ ║
J__inference_gaussian_noise_1_layer_call_and_return_conditional_losses_4856l;в8
1в.
(К%
inputs         	
p
к "-в*
#К 
0         	
Ъ Т
/__inference_gaussian_noise_1_layer_call_fn_4836_;в8
1в.
(К%
inputs         	
p 
к " К         	Т
/__inference_gaussian_noise_1_layer_call_fn_4841_;в8
1в.
(К%
inputs         	
p
к " К         	║
J__inference_gaussian_noise_2_layer_call_and_return_conditional_losses_5021l;в8
1в.
(К%
inputs         
p 
к "-в*
#К 
0         
Ъ ║
J__inference_gaussian_noise_2_layer_call_and_return_conditional_losses_5032l;в8
1в.
(К%
inputs         
p
к "-в*
#К 
0         
Ъ Т
/__inference_gaussian_noise_2_layer_call_fn_5012_;в8
1в.
(К%
inputs         
p 
к " К         Т
/__inference_gaussian_noise_2_layer_call_fn_5017_;в8
1в.
(К%
inputs         
p
к " К         ╕
H__inference_gaussian_noise_layer_call_and_return_conditional_losses_4669l;в8
1в.
(К%
inputs         
p 
к "-в*
#К 
0         
Ъ ╕
H__inference_gaussian_noise_layer_call_and_return_conditional_losses_4680l;в8
1в.
(К%
inputs         
p
к "-в*
#К 
0         
Ъ Р
-__inference_gaussian_noise_layer_call_fn_4660_;в8
1в.
(К%
inputs         
p 
к " К         Р
-__inference_gaussian_noise_layer_call_fn_4665_;в8
1в.
(К%
inputs         
p
к " К         ╜
M__inference_layer_normalization_layer_call_and_return_conditional_losses_4494lHI7в4
-в*
(К%
inputs         x'
к "-в*
#К 
0         x'
Ъ Х
2__inference_layer_normalization_layer_call_fn_4468_HI7в4
-в*
(К%
inputs         x'
к " К         x'9
__inference_loss_fn_0_5180Nв

в 
к "К 9
__inference_loss_fn_1_5191qв

в 
к "К :
__inference_loss_fn_2_5202Рв

в 
к "К :
__inference_loss_fn_3_5213╝в

в 
к "К ъ
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_4650ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ │
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_4655h7в4
-в*
(К%
inputs         
к "-в*
#К 
0         
Ъ ┬
,__inference_max_pooling2d_layer_call_fn_4640СRвO
HвE
CК@
inputs4                                    
к ";К84                                    Л
,__inference_max_pooling2d_layer_call_fn_4645[7в4
-в*
(К%
inputs         
к " К         ╥
R__inference_photoreceptor_rods_reike_layer_call_and_return_conditional_losses_4440|()*+,-12./456789:;<=035в2
+в(
&К#
inputs         ┤Т	
к "+в(
!К
0         ┤Т	
Ъ к
7__inference_photoreceptor_rods_reike_layer_call_fn_4379o()*+,-12./456789:;<=035в2
+в(
&К#
inputs         ┤Т	
к "К         ┤Т	о
C__inference_reshape_1_layer_call_and_return_conditional_losses_4459g5в2
+в(
&К#
inputs         ┤Т	
к ".в+
$К!
0         ┤'
Ъ Ж
(__inference_reshape_1_layer_call_fn_4445Z5в2
+в(
&К#
inputs         ┤Т	
к "!К         ┤'и
C__inference_reshape_2_layer_call_and_return_conditional_losses_4635a0в-
&в#
!К
inputs         ╨*
к "-в*
#К 
0         
Ъ А
(__inference_reshape_2_layer_call_fn_4621T0в-
&в#
!К
inputs         ╨*
к " К         и
C__inference_reshape_3_layer_call_and_return_conditional_losses_4831a0в-
&в#
!К
inputs         ╨
к "-в*
#К 
0         	
Ъ А
(__inference_reshape_3_layer_call_fn_4817T0в-
&в#
!К
inputs         ╨
к " К         	з
C__inference_reshape_4_layer_call_and_return_conditional_losses_5007`/в,
%в"
 К
inputs         Z
к "-в*
#К 
0         
Ъ 
(__inference_reshape_4_layer_call_fn_4993S/в,
%в"
 К
inputs         Z
к " К         м
A__inference_reshape_layer_call_and_return_conditional_losses_4330g8в5
.в+
)К&
inputs         ┤'
к "+в(
!К
0         ┤Т	
Ъ Д
&__inference_reshape_layer_call_fn_4317Z8в5
.в+
)К&
inputs         ┤'
к "К         ┤Т	ш
"__inference_signature_wrapper_3483┴<()*+,-12./456789:;<=03HINO\Y[Zqr|~}РСЮЫЭЬ╖┤╢╡╝╜DвA
в 
:к7
5
input_1*К'
input_1         ┤'";к8
6
activation_3&К#
activation_3         %
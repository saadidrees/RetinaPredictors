��
��
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
2	��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
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
�
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
�
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
�
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
delete_old_dirsbool(�
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
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
@
ReadVariableOp
resource
value"dtype"
dtypetype�
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
list(type)(0�
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
list(type)(0�
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

2	�
�
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
executor_typestring ��
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
�
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
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.7.02v2.7.0-rc1-69-gc256c071bb28��
�
layer_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:x'**
shared_namelayer_normalization/gamma
�
-layer_normalization/gamma/Read/ReadVariableOpReadVariableOplayer_normalization/gamma*"
_output_shapes
:x'*
dtype0
�
layer_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:x'*)
shared_namelayer_normalization/beta
�
,layer_normalization/beta/Read/ReadVariableOpReadVariableOplayer_normalization/beta*"
_output_shapes
:x'*
dtype0
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:		x*
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:		x*
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
:*
dtype0
�
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:*
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
:*
dtype0
�
batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�**
shared_namebatch_normalization/gamma
�
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes	
:�*
dtype0
�
batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*)
shared_namebatch_normalization/beta
�
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes	
:�*
dtype0
�
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*0
shared_name!batch_normalization/moving_mean
�
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes	
:�*
dtype0
�
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*4
shared_name%#batch_normalization/moving_variance
�
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes	
:�*
dtype0
�
conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_2/kernel
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*&
_output_shapes
:*
dtype0
r
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_2/bias
k
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes
:*
dtype0
�
batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*,
shared_namebatch_normalization_1/gamma
�
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes
:Z*
dtype0
�
batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*+
shared_namebatch_normalization_1/beta
�
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes
:Z*
dtype0
�
!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*2
shared_name#!batch_normalization_1/moving_mean
�
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes
:Z*
dtype0
�
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*6
shared_name'%batch_normalization_1/moving_variance
�
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes
:Z*
dtype0
�
batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*,
shared_namebatch_normalization_2/gamma
�
/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes
:Z*
dtype0
�
batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*+
shared_namebatch_normalization_2/beta
�
.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes
:Z*
dtype0
�
!batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*2
shared_name#!batch_normalization_2/moving_mean
�
5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes
:Z*
dtype0
�
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*6
shared_name'%batch_normalization_2/moving_variance
�
9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
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

NoOpNoOp
�V
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�V
value�UB�U B�U
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
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
layer_with_weights-4
layer-12
layer-13
layer_with_weights-5
layer-14
layer-15
layer-16
layer-17
layer-18
layer_with_weights-6
layer-19
layer_with_weights-7
layer-20
layer-21
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
 
q
axis
	gamma
beta
 	variables
!trainable_variables
"regularization_losses
#	keras_api
h

$kernel
%bias
&	variables
'trainable_variables
(regularization_losses
)	keras_api
R
*	variables
+trainable_variables
,regularization_losses
-	keras_api
R
.	variables
/trainable_variables
0regularization_losses
1	keras_api
R
2	variables
3trainable_variables
4regularization_losses
5	keras_api
h

6kernel
7bias
8	variables
9trainable_variables
:regularization_losses
;	keras_api
R
<	variables
=trainable_variables
>regularization_losses
?	keras_api
�
@axis
	Agamma
Bbeta
Cmoving_mean
Dmoving_variance
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
R
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
R
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
R
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
h

Ukernel
Vbias
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
R
[	variables
\trainable_variables
]regularization_losses
^	keras_api
�
_axis
	`gamma
abeta
bmoving_mean
cmoving_variance
d	variables
etrainable_variables
fregularization_losses
g	keras_api
R
h	variables
itrainable_variables
jregularization_losses
k	keras_api
R
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
R
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
R
t	variables
utrainable_variables
vregularization_losses
w	keras_api
�
xaxis
	ygamma
zbeta
{moving_mean
|moving_variance
}	variables
~trainable_variables
regularization_losses
�	keras_api
n
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
V
�	variables
�trainable_variables
�regularization_losses
�	keras_api
 
�
0
1
$2
%3
64
75
A6
B7
C8
D9
U10
V11
`12
a13
b14
c15
y16
z17
{18
|19
�20
�21
x
0
1
$2
%3
64
75
A6
B7
U8
V9
`10
a11
y12
z13
�14
�15
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
 
 
db
VARIABLE_VALUElayer_normalization/gamma5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUElayer_normalization/beta4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
 	variables
!trainable_variables
"regularization_losses
YW
VARIABLE_VALUEconv2d/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

$0
%1

$0
%1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
&	variables
'trainable_variables
(regularization_losses
 
 
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
*	variables
+trainable_variables
,regularization_losses
 
 
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
.	variables
/trainable_variables
0regularization_losses
 
 
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
2	variables
3trainable_variables
4regularization_losses
[Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

60
71

60
71
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
8	variables
9trainable_variables
:regularization_losses
 
 
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
<	variables
=trainable_variables
>regularization_losses
 
db
VARIABLE_VALUEbatch_normalization/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbatch_normalization/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEbatch_normalization/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE#batch_normalization/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

A0
B1
C2
D3

A0
B1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
 
 
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
 
 
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
 
 
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
[Y
VARIABLE_VALUEconv2d_2/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_2/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

U0
V1

U0
V1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
 
 
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
[	variables
\trainable_variables
]regularization_losses
 
fd
VARIABLE_VALUEbatch_normalization_1/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_1/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_1/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_1/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

`0
a1
b2
c3

`0
a1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
d	variables
etrainable_variables
fregularization_losses
 
 
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
h	variables
itrainable_variables
jregularization_losses
 
 
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
l	variables
mtrainable_variables
nregularization_losses
 
 
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
p	variables
qtrainable_variables
rregularization_losses
 
 
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
t	variables
utrainable_variables
vregularization_losses
 
fd
VARIABLE_VALUEbatch_normalization_2/gamma5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_2/beta4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_2/moving_mean;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_2/moving_variance?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

y0
z1
{2
|3

y0
z1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
}	variables
~trainable_variables
regularization_losses
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

�0
�1

�0
�1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
 
 
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
*
C0
D1
b2
c3
{4
|5
�
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

�0
�1
�2
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
 
 
 

C0
D1
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
b0
c1
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
{0
|1
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
I

�total

�count
�
_fn_kwargs
�	variables
�	keras_api
I

�total

�count
�
_fn_kwargs
�	variables
�	keras_api
I

�total

�count
�
_fn_kwargs
�	variables
�	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 

�0
�1

�	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

�0
�1

�	variables
QO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE
 

�0
�1

�	variables
�
serving_default_input_1Placeholder*/
_output_shapes
:���������x'*
dtype0*$
shape:���������x'
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1layer_normalization/gammalayer_normalization/betaconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/bias#batch_normalization/moving_variancebatch_normalization/gammabatch_normalization/moving_meanbatch_normalization/betaconv2d_2/kernelconv2d_2/bias%batch_normalization_1/moving_variancebatch_normalization_1/gamma!batch_normalization_1/moving_meanbatch_normalization_1/beta%batch_normalization_2/moving_variancebatch_normalization_2/gamma!batch_normalization_2/moving_meanbatch_normalization_2/betadense/kernel
dense/bias*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������%*8
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *+
f&R$
"__inference_signature_wrapper_2612
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename-layer_normalization/gamma/Read/ReadVariableOp,layer_normalization/beta/Read/ReadVariableOp!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp/batch_normalization_2/gamma/Read/ReadVariableOp.batch_normalization_2/beta/Read/ReadVariableOp5batch_normalization_2/moving_mean/Read/ReadVariableOp9batch_normalization_2/moving_variance/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOpConst*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *&
f!R
__inference__traced_save_3790
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamelayer_normalization/gammalayer_normalization/betaconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv2d_2/kernelconv2d_2/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_variancebatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_variancedense/kernel
dense/biastotalcounttotal_1count_1total_2count_2*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *)
f$R"
 __inference__traced_restore_3884��
�
]
A__inference_flatten_layer_call_and_return_conditional_losses_1494

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������	:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
��
�
E__inference_CNN_2D_NORM_layer_call_and_return_conditional_losses_2831

inputsZ
Dlayer_normalization_reshape_readvariableop_layer_normalization_gamma:x'[
Elayer_normalization_reshape_1_readvariableop_layer_normalization_beta:x'D
*conv2d_conv2d_readvariableop_conv2d_kernel:		x7
)conv2d_biasadd_readvariableop_conv2d_bias:H
.conv2d_1_conv2d_readvariableop_conv2d_1_kernel:;
-conv2d_1_biasadd_readvariableop_conv2d_1_bias:_
Pbatch_normalization_batchnorm_readvariableop_batch_normalization_moving_variance:	�Y
Jbatch_normalization_batchnorm_mul_readvariableop_batch_normalization_gamma:	�]
Nbatch_normalization_batchnorm_readvariableop_1_batch_normalization_moving_mean:	�V
Gbatch_normalization_batchnorm_readvariableop_2_batch_normalization_beta:	�H
.conv2d_2_conv2d_readvariableop_conv2d_2_kernel:;
-conv2d_2_biasadd_readvariableop_conv2d_2_bias:b
Tbatch_normalization_1_batchnorm_readvariableop_batch_normalization_1_moving_variance:Z\
Nbatch_normalization_1_batchnorm_mul_readvariableop_batch_normalization_1_gamma:Z`
Rbatch_normalization_1_batchnorm_readvariableop_1_batch_normalization_1_moving_mean:ZY
Kbatch_normalization_1_batchnorm_readvariableop_2_batch_normalization_1_beta:Zb
Tbatch_normalization_2_batchnorm_readvariableop_batch_normalization_2_moving_variance:Z\
Nbatch_normalization_2_batchnorm_mul_readvariableop_batch_normalization_2_gamma:Z`
Rbatch_normalization_2_batchnorm_readvariableop_1_batch_normalization_2_moving_mean:ZY
Kbatch_normalization_2_batchnorm_readvariableop_2_batch_normalization_2_beta:Z:
(dense_matmul_readvariableop_dense_kernel:Z%5
'dense_biasadd_readvariableop_dense_bias:%
identity��,batch_normalization/batchnorm/ReadVariableOp�.batch_normalization/batchnorm/ReadVariableOp_1�.batch_normalization/batchnorm/ReadVariableOp_2�0batch_normalization/batchnorm/mul/ReadVariableOp�.batch_normalization_1/batchnorm/ReadVariableOp�0batch_normalization_1/batchnorm/ReadVariableOp_1�0batch_normalization_1/batchnorm/ReadVariableOp_2�2batch_normalization_1/batchnorm/mul/ReadVariableOp�.batch_normalization_2/batchnorm/ReadVariableOp�0batch_normalization_2/batchnorm/ReadVariableOp_1�0batch_normalization_2/batchnorm/ReadVariableOp_2�2batch_normalization_2/batchnorm/mul/ReadVariableOp�conv2d/BiasAdd/ReadVariableOp�conv2d/Conv2D/ReadVariableOp�/conv2d/kernel/Regularizer/Square/ReadVariableOp�conv2d_1/BiasAdd/ReadVariableOp�conv2d_1/Conv2D/ReadVariableOp�1conv2d_1/kernel/Regularizer/Square/ReadVariableOp�conv2d_2/BiasAdd/ReadVariableOp�conv2d_2/Conv2D/ReadVariableOp�1conv2d_2/kernel/Regularizer/Square/ReadVariableOp�dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�.dense/kernel/Regularizer/Square/ReadVariableOp�*layer_normalization/Reshape/ReadVariableOp�,layer_normalization/Reshape_1/ReadVariableOp�
2layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         �
 layer_normalization/moments/meanMeaninputs;layer_normalization/moments/mean/reduction_indices:output:0*
T0*/
_output_shapes
:���������*
	keep_dims(�
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*/
_output_shapes
:����������
-layer_normalization/moments/SquaredDifferenceSquaredDifferenceinputs1layer_normalization/moments/StopGradient:output:0*
T0*/
_output_shapes
:���������x'�
6layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         �
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
T0*/
_output_shapes
:���������*
	keep_dims(�
*layer_normalization/Reshape/ReadVariableOpReadVariableOpDlayer_normalization_reshape_readvariableop_layer_normalization_gamma*"
_output_shapes
:x'*
dtype0z
!layer_normalization/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   x      '   �
layer_normalization/ReshapeReshape2layer_normalization/Reshape/ReadVariableOp:value:0*layer_normalization/Reshape/shape:output:0*
T0*&
_output_shapes
:x'�
,layer_normalization/Reshape_1/ReadVariableOpReadVariableOpElayer_normalization_reshape_1_readvariableop_layer_normalization_beta*"
_output_shapes
:x'*
dtype0|
#layer_normalization/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   x      '   �
layer_normalization/Reshape_1Reshape4layer_normalization/Reshape_1/ReadVariableOp:value:0,layer_normalization/Reshape_1/shape:output:0*
T0*&
_output_shapes
:x'h
#layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*/
_output_shapes
:����������
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*/
_output_shapes
:����������
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:0$layer_normalization/Reshape:output:0*
T0*/
_output_shapes
:���������x'�
#layer_normalization/batchnorm/mul_1Mulinputs%layer_normalization/batchnorm/mul:z:0*
T0*/
_output_shapes
:���������x'�
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*/
_output_shapes
:���������x'�
!layer_normalization/batchnorm/subSub&layer_normalization/Reshape_1:output:0'layer_normalization/batchnorm/mul_2:z:0*
T0*/
_output_shapes
:���������x'�
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*/
_output_shapes
:���������x'�
conv2d/Conv2D/ReadVariableOpReadVariableOp*conv2d_conv2d_readvariableop_conv2d_kernel*&
_output_shapes
:		x*
dtype0�
conv2d/Conv2DConv2D'layer_normalization/batchnorm/add_1:z:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
data_formatNCHW*
paddingVALID*
strides
�
conv2d/BiasAdd/ReadVariableOpReadVariableOp)conv2d_biasadd_readvariableop_conv2d_bias*
_output_shapes
:*
dtype0�
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
data_formatNCHW�
max_pooling2d/MaxPoolMaxPoolconv2d/BiasAdd:output:0*/
_output_shapes
:���������*
data_formatNCHW*
ksize
*
paddingVALID*
strides
q
activation/ReluRelumax_pooling2d/MaxPool:output:0*
T0*/
_output_shapes
:����������
conv2d_1/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_conv2d_readvariableop_conv2d_1_kernel*&
_output_shapes
:*
dtype0�
conv2d_1/Conv2DConv2Dactivation/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������	*
data_formatNCHW*
paddingVALID*
strides
�
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp-conv2d_1_biasadd_readvariableop_conv2d_1_bias*
_output_shapes
:*
dtype0�
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������	*
data_formatNCHW^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  �
flatten/ReshapeReshapeconv2d_1/BiasAdd:output:0flatten/Const:output:0*
T0*(
_output_shapes
:�����������
,batch_normalization/batchnorm/ReadVariableOpReadVariableOpPbatch_normalization_batchnorm_readvariableop_batch_normalization_moving_variance*
_output_shapes	
:�*
dtype0h
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
!batch_normalization/batchnorm/addAddV24batch_normalization/batchnorm/ReadVariableOp:value:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�y
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:��
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOpJbatch_normalization_batchnorm_mul_readvariableop_batch_normalization_gamma*
_output_shapes	
:�*
dtype0�
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
#batch_normalization/batchnorm/mul_1Mulflatten/Reshape:output:0%batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
.batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOpNbatch_normalization_batchnorm_readvariableop_1_batch_normalization_moving_mean*
_output_shapes	
:�*
dtype0�
#batch_normalization/batchnorm/mul_2Mul6batch_normalization/batchnorm/ReadVariableOp_1:value:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
.batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOpGbatch_normalization_batchnorm_readvariableop_2_batch_normalization_beta*
_output_shapes	
:�*
dtype0�
!batch_normalization/batchnorm/subSub6batch_normalization/batchnorm/ReadVariableOp_2:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������d
reshape/ShapeShape'batch_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:e
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
valueB:�
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Y
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Y
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :	�
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
reshape/ReshapeReshape'batch_normalization/batchnorm/add_1:z:0reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:���������	m
activation_1/ReluRelureshape/Reshape:output:0*
T0*/
_output_shapes
:���������	�
conv2d_2/Conv2D/ReadVariableOpReadVariableOp.conv2d_2_conv2d_readvariableop_conv2d_2_kernel*&
_output_shapes
:*
dtype0�
conv2d_2/Conv2DConv2Dactivation_1/Relu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
data_formatNCHW*
paddingVALID*
strides
�
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp-conv2d_2_biasadd_readvariableop_conv2d_2_bias*
_output_shapes
:*
dtype0�
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
data_formatNCHW`
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
flatten_1/ReshapeReshapeconv2d_2/BiasAdd:output:0flatten_1/Const:output:0*
T0*'
_output_shapes
:���������Z�
.batch_normalization_1/batchnorm/ReadVariableOpReadVariableOpTbatch_normalization_1_batchnorm_readvariableop_batch_normalization_1_moving_variance*
_output_shapes
:Z*
dtype0j
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
#batch_normalization_1/batchnorm/addAddV26batch_normalization_1/batchnorm/ReadVariableOp:value:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes
:Z|
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
:Z�
2batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpNbatch_normalization_1_batchnorm_mul_readvariableop_batch_normalization_1_gamma*
_output_shapes
:Z*
dtype0�
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:0:batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Z�
%batch_normalization_1/batchnorm/mul_1Mulflatten_1/Reshape:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������Z�
0batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOpRbatch_normalization_1_batchnorm_readvariableop_1_batch_normalization_1_moving_mean*
_output_shapes
:Z*
dtype0�
%batch_normalization_1/batchnorm/mul_2Mul8batch_normalization_1/batchnorm/ReadVariableOp_1:value:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
:Z�
0batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOpKbatch_normalization_1_batchnorm_readvariableop_2_batch_normalization_1_beta*
_output_shapes
:Z*
dtype0�
#batch_normalization_1/batchnorm/subSub8batch_normalization_1/batchnorm/ReadVariableOp_2:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Z�
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������Zh
reshape_1/ShapeShape)batch_normalization_1/batchnorm/add_1:z:0*
T0*
_output_shapes
:g
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
valueB:�
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0"reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
reshape_1/ReshapeReshape)batch_normalization_1/batchnorm/add_1:z:0 reshape_1/Reshape/shape:output:0*
T0*/
_output_shapes
:���������o
activation_2/ReluRelureshape_1/Reshape:output:0*
T0*/
_output_shapes
:���������`
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
flatten_2/ReshapeReshapeactivation_2/Relu:activations:0flatten_2/Const:output:0*
T0*'
_output_shapes
:���������Z�
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOpTbatch_normalization_2_batchnorm_readvariableop_batch_normalization_2_moving_variance*
_output_shapes
:Z*
dtype0j
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
#batch_normalization_2/batchnorm/addAddV26batch_normalization_2/batchnorm/ReadVariableOp:value:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:Z|
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:Z�
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpNbatch_normalization_2_batchnorm_mul_readvariableop_batch_normalization_2_gamma*
_output_shapes
:Z*
dtype0�
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Z�
%batch_normalization_2/batchnorm/mul_1Mulflatten_2/Reshape:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������Z�
0batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOpRbatch_normalization_2_batchnorm_readvariableop_1_batch_normalization_2_moving_mean*
_output_shapes
:Z*
dtype0�
%batch_normalization_2/batchnorm/mul_2Mul8batch_normalization_2/batchnorm/ReadVariableOp_1:value:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:Z�
0batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOpKbatch_normalization_2_batchnorm_readvariableop_2_batch_normalization_2_beta*
_output_shapes
:Z*
dtype0�
#batch_normalization_2/batchnorm/subSub8batch_normalization_2/batchnorm/ReadVariableOp_2:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Z�
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������Z�
dense/MatMul/ReadVariableOpReadVariableOp(dense_matmul_readvariableop_dense_kernel*
_output_shapes

:Z%*
dtype0�
dense/MatMulMatMul)batch_normalization_2/batchnorm/add_1:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������%�
dense/BiasAdd/ReadVariableOpReadVariableOp'dense_biasadd_readvariableop_dense_bias*
_output_shapes
:%*
dtype0�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������%n
dense/ActivityRegularizer/AbsAbsdense/BiasAdd:output:0*
T0*'
_output_shapes
:���������%p
dense/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense/ActivityRegularizer/SumSum!dense/ActivityRegularizer/Abs:y:0(dense/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: d
dense/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
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
valueB:�
'dense/ActivityRegularizer/strided_sliceStridedSlice(dense/ActivityRegularizer/Shape:output:06dense/ActivityRegularizer/strided_slice/stack:output:08dense/ActivityRegularizer/strided_slice/stack_1:output:08dense/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
dense/ActivityRegularizer/CastCast0dense/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: �
!dense/ActivityRegularizer/truedivRealDiv!dense/ActivityRegularizer/mul:z:0"dense/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: k
activation_3/SoftplusSoftplusdense/BiasAdd:output:0*
T0*'
_output_shapes
:���������%�
/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp*conv2d_conv2d_readvariableop_conv2d_kernel*&
_output_shapes
:		x*
dtype0�
 conv2d/kernel/Regularizer/SquareSquare7conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		xx
conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d/kernel/Regularizer/SumSum$conv2d/kernel/Regularizer/Square:y:0(conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
conv2d/kernel/Regularizer/mulMul(conv2d/kernel/Regularizer/mul/x:output:0&conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp.conv2d_1_conv2d_readvariableop_conv2d_1_kernel*&
_output_shapes
:*
dtype0�
"conv2d_1/kernel/Regularizer/SquareSquare9conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
!conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_1/kernel/Regularizer/SumSum&conv2d_1/kernel/Regularizer/Square:y:0*conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
conv2d_1/kernel/Regularizer/mulMul*conv2d_1/kernel/Regularizer/mul/x:output:0(conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp.conv2d_2_conv2d_readvariableop_conv2d_2_kernel*&
_output_shapes
:*
dtype0�
"conv2d_2/kernel/Regularizer/SquareSquare9conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
!conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_2/kernel/Regularizer/SumSum&conv2d_2/kernel/Regularizer/Square:y:0*conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
conv2d_2/kernel/Regularizer/mulMul*conv2d_2/kernel/Regularizer/mul/x:output:0(conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(dense_matmul_readvariableop_dense_kernel*
_output_shapes

:Z%*
dtype0�
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:Z%o
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: r
IdentityIdentity#activation_3/Softplus:activations:0^NoOp*
T0*'
_output_shapes
:���������%�	
NoOpNoOp-^batch_normalization/batchnorm/ReadVariableOp/^batch_normalization/batchnorm/ReadVariableOp_1/^batch_normalization/batchnorm/ReadVariableOp_21^batch_normalization/batchnorm/mul/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp1^batch_normalization_1/batchnorm/ReadVariableOp_11^batch_normalization_1/batchnorm/ReadVariableOp_23^batch_normalization_1/batchnorm/mul/ReadVariableOp/^batch_normalization_2/batchnorm/ReadVariableOp1^batch_normalization_2/batchnorm/ReadVariableOp_11^batch_normalization_2/batchnorm/ReadVariableOp_23^batch_normalization_2/batchnorm/mul/ReadVariableOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp0^conv2d/kernel/Regularizer/Square/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp2^conv2d_1/kernel/Regularizer/Square/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp2^conv2d_2/kernel/Regularizer/Square/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp+^layer_normalization/Reshape/ReadVariableOp-^layer_normalization/Reshape_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*Z
_input_shapesI
G:���������x': : : : : : : : : : : : : : : : : : : : : : 2\
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
2batch_normalization_2/batchnorm/mul/ReadVariableOp2batch_normalization_2/batchnorm/mul/ReadVariableOp2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2b
/conv2d/kernel/Regularizer/Square/ReadVariableOp/conv2d/kernel/Regularizer/Square/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2f
1conv2d_1/kernel/Regularizer/Square/ReadVariableOp1conv2d_1/kernel/Regularizer/Square/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2f
1conv2d_2/kernel/Regularizer/Square/ReadVariableOp1conv2d_2/kernel/Regularizer/Square/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2X
*layer_normalization/Reshape/ReadVariableOp*layer_normalization/Reshape/ReadVariableOp2\
,layer_normalization/Reshape_1/ReadVariableOp,layer_normalization/Reshape_1/ReadVariableOp:W S
/
_output_shapes
:���������x'
 
_user_specified_nameinputs
��
�
E__inference_CNN_2D_NORM_layer_call_and_return_conditional_losses_2559
input_1C
-layer_normalization_layer_normalization_gamma:x'B
,layer_normalization_layer_normalization_beta:x'.
conv2d_conv2d_kernel:		x 
conv2d_conv2d_bias:2
conv2d_1_conv2d_1_kernel:$
conv2d_1_conv2d_1_bias:B
3batch_normalization_batch_normalization_moving_mean:	�F
7batch_normalization_batch_normalization_moving_variance:	�<
-batch_normalization_batch_normalization_gamma:	�;
,batch_normalization_batch_normalization_beta:	�2
conv2d_2_conv2d_2_kernel:$
conv2d_2_conv2d_2_bias:E
7batch_normalization_1_batch_normalization_1_moving_mean:ZI
;batch_normalization_1_batch_normalization_1_moving_variance:Z?
1batch_normalization_1_batch_normalization_1_gamma:Z>
0batch_normalization_1_batch_normalization_1_beta:ZE
7batch_normalization_2_batch_normalization_2_moving_mean:ZI
;batch_normalization_2_batch_normalization_2_moving_variance:Z?
1batch_normalization_2_batch_normalization_2_gamma:Z>
0batch_normalization_2_batch_normalization_2_beta:Z$
dense_dense_kernel:Z%
dense_dense_bias:%
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�-batch_normalization_2/StatefulPartitionedCall�conv2d/StatefulPartitionedCall�/conv2d/kernel/Regularizer/Square/ReadVariableOp� conv2d_1/StatefulPartitionedCall�1conv2d_1/kernel/Regularizer/Square/ReadVariableOp� conv2d_2/StatefulPartitionedCall�1conv2d_2/kernel/Regularizer/Square/ReadVariableOp�dense/StatefulPartitionedCall�.dense/kernel/Regularizer/Square/ReadVariableOp�&gaussian_noise/StatefulPartitionedCall�(gaussian_noise_1/StatefulPartitionedCall�(gaussian_noise_2/StatefulPartitionedCall�+layer_normalization/StatefulPartitionedCall�
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCallinput_1-layer_normalization_layer_normalization_gamma,layer_normalization_layer_normalization_beta*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������x'*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *V
fQRO
M__inference_layer_normalization_layer_call_and_return_conditional_losses_1425�
conv2d/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0conv2d_conv2d_kernelconv2d_conv2d_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_1445�
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_1453�
&gaussian_noise/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_gaussian_noise_layer_call_and_return_conditional_losses_1997�
activation/PartitionedCallPartitionedCall/gaussian_noise/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_activation_layer_call_and_return_conditional_losses_1466�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0conv2d_1_conv2d_1_kernelconv2d_1_conv2d_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_1484�
flatten/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_1494�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:03batch_normalization_batch_normalization_moving_mean7batch_normalization_batch_normalization_moving_variance-batch_normalization_batch_normalization_gamma,batch_normalization_batch_normalization_beta*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_1062�
reshape/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_reshape_layer_call_and_return_conditional_losses_1515�
(gaussian_noise_1/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0'^gaussian_noise/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *S
fNRL
J__inference_gaussian_noise_1_layer_call_and_return_conditional_losses_1893�
activation_1/PartitionedCallPartitionedCall1gaussian_noise_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_activation_1_layer_call_and_return_conditional_losses_1528�
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0conv2d_2_conv2d_2_kernelconv2d_2_conv2d_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_1546�
flatten_1/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������Z* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_flatten_1_layer_call_and_return_conditional_losses_1556�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:07batch_normalization_1_batch_normalization_1_moving_mean;batch_normalization_1_batch_normalization_1_moving_variance1batch_normalization_1_batch_normalization_1_gamma0batch_normalization_1_batch_normalization_1_beta*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������Z*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1190�
reshape_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_reshape_1_layer_call_and_return_conditional_losses_1577�
(gaussian_noise_2/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0)^gaussian_noise_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *S
fNRL
J__inference_gaussian_noise_2_layer_call_and_return_conditional_losses_1789�
activation_2/PartitionedCallPartitionedCall1gaussian_noise_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_activation_2_layer_call_and_return_conditional_losses_1590�
flatten_2/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������Z* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_flatten_2_layer_call_and_return_conditional_losses_1598�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:07batch_normalization_2_batch_normalization_2_moving_mean;batch_normalization_2_batch_normalization_2_moving_variance1batch_normalization_2_batch_normalization_2_gamma0batch_normalization_2_batch_normalization_2_beta*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������Z*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1318�
dense/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0dense_dense_kerneldense_dense_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������%*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_1621�
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
 *2
config_proto" 

CPU

GPU2 *0J 8� *4
f/R-
+__inference_dense_activity_regularizer_1636u
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
valueB:�
'dense/ActivityRegularizer/strided_sliceStridedSlice(dense/ActivityRegularizer/Shape:output:06dense/ActivityRegularizer/strided_slice/stack:output:08dense/ActivityRegularizer/strided_slice/stack_1:output:08dense/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
dense/ActivityRegularizer/CastCast0dense/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: �
!dense/ActivityRegularizer/truedivRealDiv2dense/ActivityRegularizer/PartitionedCall:output:0"dense/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: �
activation_3/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������%* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_activation_3_layer_call_and_return_conditional_losses_1650�
/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_conv2d_kernel*&
_output_shapes
:		x*
dtype0�
 conv2d/kernel/Regularizer/SquareSquare7conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		xx
conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d/kernel/Regularizer/SumSum$conv2d/kernel/Regularizer/Square:y:0(conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
conv2d/kernel/Regularizer/mulMul(conv2d/kernel/Regularizer/mul/x:output:0&conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_1_conv2d_1_kernel*&
_output_shapes
:*
dtype0�
"conv2d_1/kernel/Regularizer/SquareSquare9conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
!conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_1/kernel/Regularizer/SumSum&conv2d_1/kernel/Regularizer/Square:y:0*conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
conv2d_1/kernel/Regularizer/mulMul*conv2d_1/kernel/Regularizer/mul/x:output:0(conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_2_conv2d_2_kernel*&
_output_shapes
:*
dtype0�
"conv2d_2/kernel/Regularizer/SquareSquare9conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
!conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_2/kernel/Regularizer/SumSum&conv2d_2/kernel/Regularizer/Square:y:0*conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
conv2d_2/kernel/Regularizer/mulMul*conv2d_2/kernel/Regularizer/mul/x:output:0(conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_dense_kernel*
_output_shapes

:Z%*
dtype0�
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:Z%o
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: t
IdentityIdentity%activation_3/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������%�
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall^conv2d/StatefulPartitionedCall0^conv2d/kernel/Regularizer/Square/ReadVariableOp!^conv2d_1/StatefulPartitionedCall2^conv2d_1/kernel/Regularizer/Square/ReadVariableOp!^conv2d_2/StatefulPartitionedCall2^conv2d_2/kernel/Regularizer/Square/ReadVariableOp^dense/StatefulPartitionedCall/^dense/kernel/Regularizer/Square/ReadVariableOp'^gaussian_noise/StatefulPartitionedCall)^gaussian_noise_1/StatefulPartitionedCall)^gaussian_noise_2/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*Z
_input_shapesI
G:���������x': : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2b
/conv2d/kernel/Regularizer/Square/ReadVariableOp/conv2d/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2f
1conv2d_1/kernel/Regularizer/Square/ReadVariableOp1conv2d_1/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2f
1conv2d_2/kernel/Regularizer/Square/ReadVariableOp1conv2d_2/kernel/Regularizer/Square/ReadVariableOp2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2P
&gaussian_noise/StatefulPartitionedCall&gaussian_noise/StatefulPartitionedCall2T
(gaussian_noise_1/StatefulPartitionedCall(gaussian_noise_1/StatefulPartitionedCall2T
(gaussian_noise_2/StatefulPartitionedCall(gaussian_noise_2/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall:X T
/
_output_shapes
:���������x'
!
_user_specified_name	input_1
�	
i
J__inference_gaussian_noise_2_layer_call_and_return_conditional_losses_1789

inputs
identity�;
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
 *���=�
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*/
_output_shapes
:���������*
dtype0�
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*/
_output_shapes
:����������
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*/
_output_shapes
:���������a
addAddV2inputsrandom_normal:z:0*
T0*/
_output_shapes
:���������W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
]
A__inference_flatten_layer_call_and_return_conditional_losses_3216

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*.
_input_shapes
:���������	:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
c
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_3141

inputs
identity�
MaxPoolMaxPoolinputs*/
_output_shapes
:���������*
data_formatNCHW*
ksize
*
paddingVALID*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
B
+__inference_dense_activity_regularizer_1392
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
:���������D
SumSumAbs:y:0range:output:0*
T0*
_output_shapes
: J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:I
mulMulmul/x:output:0Sum:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*
_input_shapes
::; 7

_output_shapes
:

_user_specified_namex
�
b
F__inference_activation_2_layer_call_and_return_conditional_losses_3508

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:���������b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
`
D__inference_activation_layer_call_and_return_conditional_losses_3176

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:���������b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
f
J__inference_gaussian_noise_1_layer_call_and_return_conditional_losses_1521

inputs
identityV
IdentityIdentityinputs*
T0*/
_output_shapes
:���������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������	:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
�
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1275

inputsL
>batchnorm_readvariableop_batch_normalization_2_moving_variance:ZF
8batchnorm_mul_readvariableop_batch_normalization_2_gamma:ZJ
<batchnorm_readvariableop_1_batch_normalization_2_moving_mean:ZC
5batchnorm_readvariableop_2_batch_normalization_2_beta:Z
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp�
batchnorm/ReadVariableOpReadVariableOp>batchnorm_readvariableop_batch_normalization_2_moving_variance*
_output_shapes
:Z*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:ZP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:Z�
batchnorm/mul/ReadVariableOpReadVariableOp8batchnorm_mul_readvariableop_batch_normalization_2_gamma*
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
:���������Z�
batchnorm/ReadVariableOp_1ReadVariableOp<batchnorm_readvariableop_1_batch_normalization_2_moving_mean*
_output_shapes
:Z*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:Z�
batchnorm/ReadVariableOp_2ReadVariableOp5batchnorm_readvariableop_2_batch_normalization_2_beta*
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
:���������Zb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������Z�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������Z: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������Z
 
_user_specified_nameinputs
�
�
*__inference_CNN_2D_NORM_layer_call_fn_2401
input_1/
layer_normalization_gamma:x'.
layer_normalization_beta:x''
conv2d_kernel:		x
conv2d_bias:)
conv2d_1_kernel:
conv2d_1_bias:.
batch_normalization_moving_mean:	�2
#batch_normalization_moving_variance:	�(
batch_normalization_gamma:	�'
batch_normalization_beta:	�)
conv2d_2_kernel:
conv2d_2_bias:/
!batch_normalization_1_moving_mean:Z3
%batch_normalization_1_moving_variance:Z)
batch_normalization_1_gamma:Z(
batch_normalization_1_beta:Z/
!batch_normalization_2_moving_mean:Z3
%batch_normalization_2_moving_variance:Z)
batch_normalization_2_gamma:Z(
batch_normalization_2_beta:Z
dense_kernel:Z%

dense_bias:%
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1layer_normalization_gammalayer_normalization_betaconv2d_kernelconv2d_biasconv2d_1_kernelconv2d_1_biasbatch_normalization_moving_mean#batch_normalization_moving_variancebatch_normalization_gammabatch_normalization_betaconv2d_2_kernelconv2d_2_bias!batch_normalization_1_moving_mean%batch_normalization_1_moving_variancebatch_normalization_1_gammabatch_normalization_1_beta!batch_normalization_2_moving_mean%batch_normalization_2_moving_variancebatch_normalization_2_gammabatch_normalization_2_betadense_kernel
dense_bias*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������%*2
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_CNN_2D_NORM_layer_call_and_return_conditional_losses_2191o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������%`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*Z
_input_shapesI
G:���������x': : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������x'
!
_user_specified_name	input_1
�
]
A__inference_reshape_layer_call_and_return_conditional_losses_1515

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
valueB:�
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
value	B :	�
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:���������	`
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:���������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�&
�
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3454

inputsN
@assignmovingavg_readvariableop_batch_normalization_1_moving_mean:ZT
Fassignmovingavg_1_readvariableop_batch_normalization_1_moving_variance:ZF
8batchnorm_mul_readvariableop_batch_normalization_1_gamma:ZA
3batchnorm_readvariableop_batch_normalization_1_beta:Z
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
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

:Z�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������Zl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
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
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp@assignmovingavg_readvariableop_batch_normalization_1_moving_mean*
_output_shapes
:Z*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:Zx
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:Z�
AssignMovingAvgAssignSubVariableOp@assignmovingavg_readvariableop_batch_normalization_1_moving_meanAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpFassignmovingavg_1_readvariableop_batch_normalization_1_moving_variance*
_output_shapes
:Z*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:Z~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:Z�
AssignMovingAvg_1AssignSubVariableOpFassignmovingavg_1_readvariableop_batch_normalization_1_moving_varianceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:ZP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:Z�
batchnorm/mul/ReadVariableOpReadVariableOp8batchnorm_mul_readvariableop_batch_normalization_1_gamma*
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
:���������Zh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:Z�
batchnorm/ReadVariableOpReadVariableOp3batchnorm_readvariableop_batch_normalization_1_beta*
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
:���������Zb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������Z�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*.
_input_shapes
:���������Z: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������Z
 
_user_specified_nameinputs
�	
�
4__inference_batch_normalization_2_layer_call_fn_3537

inputs/
!batch_normalization_2_moving_mean:Z3
%batch_normalization_2_moving_variance:Z)
batch_normalization_2_gamma:Z(
batch_normalization_2_beta:Z
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs!batch_normalization_2_moving_mean%batch_normalization_2_moving_variancebatch_normalization_2_gammabatch_normalization_2_beta*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������Z*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1318o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������Z`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*.
_input_shapes
:���������Z: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������Z
 
_user_specified_nameinputs
�
D
(__inference_flatten_2_layer_call_fn_3513

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������Z* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_flatten_2_layer_call_and_return_conditional_losses_1598`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������Z"
identityIdentity:output:0*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
*__inference_CNN_2D_NORM_layer_call_fn_2666

inputs/
layer_normalization_gamma:x'.
layer_normalization_beta:x''
conv2d_kernel:		x
conv2d_bias:)
conv2d_1_kernel:
conv2d_1_bias:.
batch_normalization_moving_mean:	�2
#batch_normalization_moving_variance:	�(
batch_normalization_gamma:	�'
batch_normalization_beta:	�)
conv2d_2_kernel:
conv2d_2_bias:/
!batch_normalization_1_moving_mean:Z3
%batch_normalization_1_moving_variance:Z)
batch_normalization_1_gamma:Z(
batch_normalization_1_beta:Z/
!batch_normalization_2_moving_mean:Z3
%batch_normalization_2_moving_variance:Z)
batch_normalization_2_gamma:Z(
batch_normalization_2_beta:Z
dense_kernel:Z%

dense_bias:%
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputslayer_normalization_gammalayer_normalization_betaconv2d_kernelconv2d_biasconv2d_1_kernelconv2d_1_biasbatch_normalization_moving_mean#batch_normalization_moving_variancebatch_normalization_gammabatch_normalization_betaconv2d_2_kernelconv2d_2_bias!batch_normalization_1_moving_mean%batch_normalization_1_moving_variancebatch_normalization_1_gammabatch_normalization_1_beta!batch_normalization_2_moving_mean%batch_normalization_2_moving_variancebatch_normalization_2_gammabatch_normalization_2_betadense_kernel
dense_bias*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������%*2
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_CNN_2D_NORM_layer_call_and_return_conditional_losses_2191o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������%`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*Z
_input_shapesI
G:���������x': : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������x'
 
_user_specified_nameinputs
�
b
F__inference_activation_3_layer_call_and_return_conditional_losses_3623

inputs
identityN
SoftplusSoftplusinputs*
T0*'
_output_shapes
:���������%^
IdentityIdentitySoftplus:activations:0*
T0*'
_output_shapes
:���������%"
identityIdentity:output:0*&
_input_shapes
:���������%:O K
'
_output_shapes
:���������%
 
_user_specified_nameinputs
��
�
E__inference_CNN_2D_NORM_layer_call_and_return_conditional_losses_2191

inputsC
-layer_normalization_layer_normalization_gamma:x'B
,layer_normalization_layer_normalization_beta:x'.
conv2d_conv2d_kernel:		x 
conv2d_conv2d_bias:2
conv2d_1_conv2d_1_kernel:$
conv2d_1_conv2d_1_bias:B
3batch_normalization_batch_normalization_moving_mean:	�F
7batch_normalization_batch_normalization_moving_variance:	�<
-batch_normalization_batch_normalization_gamma:	�;
,batch_normalization_batch_normalization_beta:	�2
conv2d_2_conv2d_2_kernel:$
conv2d_2_conv2d_2_bias:E
7batch_normalization_1_batch_normalization_1_moving_mean:ZI
;batch_normalization_1_batch_normalization_1_moving_variance:Z?
1batch_normalization_1_batch_normalization_1_gamma:Z>
0batch_normalization_1_batch_normalization_1_beta:ZE
7batch_normalization_2_batch_normalization_2_moving_mean:ZI
;batch_normalization_2_batch_normalization_2_moving_variance:Z?
1batch_normalization_2_batch_normalization_2_gamma:Z>
0batch_normalization_2_batch_normalization_2_beta:Z$
dense_dense_kernel:Z%
dense_dense_bias:%
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�-batch_normalization_2/StatefulPartitionedCall�conv2d/StatefulPartitionedCall�/conv2d/kernel/Regularizer/Square/ReadVariableOp� conv2d_1/StatefulPartitionedCall�1conv2d_1/kernel/Regularizer/Square/ReadVariableOp� conv2d_2/StatefulPartitionedCall�1conv2d_2/kernel/Regularizer/Square/ReadVariableOp�dense/StatefulPartitionedCall�.dense/kernel/Regularizer/Square/ReadVariableOp�&gaussian_noise/StatefulPartitionedCall�(gaussian_noise_1/StatefulPartitionedCall�(gaussian_noise_2/StatefulPartitionedCall�+layer_normalization/StatefulPartitionedCall�
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCallinputs-layer_normalization_layer_normalization_gamma,layer_normalization_layer_normalization_beta*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������x'*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *V
fQRO
M__inference_layer_normalization_layer_call_and_return_conditional_losses_1425�
conv2d/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0conv2d_conv2d_kernelconv2d_conv2d_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_1445�
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_1453�
&gaussian_noise/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_gaussian_noise_layer_call_and_return_conditional_losses_1997�
activation/PartitionedCallPartitionedCall/gaussian_noise/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_activation_layer_call_and_return_conditional_losses_1466�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0conv2d_1_conv2d_1_kernelconv2d_1_conv2d_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_1484�
flatten/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_1494�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:03batch_normalization_batch_normalization_moving_mean7batch_normalization_batch_normalization_moving_variance-batch_normalization_batch_normalization_gamma,batch_normalization_batch_normalization_beta*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_1062�
reshape/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_reshape_layer_call_and_return_conditional_losses_1515�
(gaussian_noise_1/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0'^gaussian_noise/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *S
fNRL
J__inference_gaussian_noise_1_layer_call_and_return_conditional_losses_1893�
activation_1/PartitionedCallPartitionedCall1gaussian_noise_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_activation_1_layer_call_and_return_conditional_losses_1528�
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0conv2d_2_conv2d_2_kernelconv2d_2_conv2d_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_1546�
flatten_1/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������Z* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_flatten_1_layer_call_and_return_conditional_losses_1556�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:07batch_normalization_1_batch_normalization_1_moving_mean;batch_normalization_1_batch_normalization_1_moving_variance1batch_normalization_1_batch_normalization_1_gamma0batch_normalization_1_batch_normalization_1_beta*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������Z*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1190�
reshape_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_reshape_1_layer_call_and_return_conditional_losses_1577�
(gaussian_noise_2/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0)^gaussian_noise_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *S
fNRL
J__inference_gaussian_noise_2_layer_call_and_return_conditional_losses_1789�
activation_2/PartitionedCallPartitionedCall1gaussian_noise_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_activation_2_layer_call_and_return_conditional_losses_1590�
flatten_2/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������Z* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_flatten_2_layer_call_and_return_conditional_losses_1598�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:07batch_normalization_2_batch_normalization_2_moving_mean;batch_normalization_2_batch_normalization_2_moving_variance1batch_normalization_2_batch_normalization_2_gamma0batch_normalization_2_batch_normalization_2_beta*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������Z*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1318�
dense/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0dense_dense_kerneldense_dense_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������%*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_1621�
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
 *2
config_proto" 

CPU

GPU2 *0J 8� *4
f/R-
+__inference_dense_activity_regularizer_1636u
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
valueB:�
'dense/ActivityRegularizer/strided_sliceStridedSlice(dense/ActivityRegularizer/Shape:output:06dense/ActivityRegularizer/strided_slice/stack:output:08dense/ActivityRegularizer/strided_slice/stack_1:output:08dense/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
dense/ActivityRegularizer/CastCast0dense/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: �
!dense/ActivityRegularizer/truedivRealDiv2dense/ActivityRegularizer/PartitionedCall:output:0"dense/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: �
activation_3/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������%* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_activation_3_layer_call_and_return_conditional_losses_1650�
/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_conv2d_kernel*&
_output_shapes
:		x*
dtype0�
 conv2d/kernel/Regularizer/SquareSquare7conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		xx
conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d/kernel/Regularizer/SumSum$conv2d/kernel/Regularizer/Square:y:0(conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
conv2d/kernel/Regularizer/mulMul(conv2d/kernel/Regularizer/mul/x:output:0&conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_1_conv2d_1_kernel*&
_output_shapes
:*
dtype0�
"conv2d_1/kernel/Regularizer/SquareSquare9conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
!conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_1/kernel/Regularizer/SumSum&conv2d_1/kernel/Regularizer/Square:y:0*conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
conv2d_1/kernel/Regularizer/mulMul*conv2d_1/kernel/Regularizer/mul/x:output:0(conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_2_conv2d_2_kernel*&
_output_shapes
:*
dtype0�
"conv2d_2/kernel/Regularizer/SquareSquare9conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
!conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_2/kernel/Regularizer/SumSum&conv2d_2/kernel/Regularizer/Square:y:0*conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
conv2d_2/kernel/Regularizer/mulMul*conv2d_2/kernel/Regularizer/mul/x:output:0(conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_dense_kernel*
_output_shapes

:Z%*
dtype0�
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:Z%o
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: t
IdentityIdentity%activation_3/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������%�
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall^conv2d/StatefulPartitionedCall0^conv2d/kernel/Regularizer/Square/ReadVariableOp!^conv2d_1/StatefulPartitionedCall2^conv2d_1/kernel/Regularizer/Square/ReadVariableOp!^conv2d_2/StatefulPartitionedCall2^conv2d_2/kernel/Regularizer/Square/ReadVariableOp^dense/StatefulPartitionedCall/^dense/kernel/Regularizer/Square/ReadVariableOp'^gaussian_noise/StatefulPartitionedCall)^gaussian_noise_1/StatefulPartitionedCall)^gaussian_noise_2/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������x': : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2b
/conv2d/kernel/Regularizer/Square/ReadVariableOp/conv2d/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2f
1conv2d_1/kernel/Regularizer/Square/ReadVariableOp1conv2d_1/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2f
1conv2d_2/kernel/Regularizer/Square/ReadVariableOp1conv2d_2/kernel/Regularizer/Square/ReadVariableOp2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2P
&gaussian_noise/StatefulPartitionedCall&gaussian_noise/StatefulPartitionedCall2T
(gaussian_noise_1/StatefulPartitionedCall(gaussian_noise_1/StatefulPartitionedCall2T
(gaussian_noise_2/StatefulPartitionedCall(gaussian_noise_2/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall:W S
/
_output_shapes
:���������x'
 
_user_specified_nameinputs
�
I
-__inference_gaussian_noise_layer_call_fn_3146

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_gaussian_noise_layer_call_and_return_conditional_losses_1459h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
4__inference_batch_normalization_2_layer_call_fn_3528

inputs3
%batch_normalization_2_moving_variance:Z)
batch_normalization_2_gamma:Z/
!batch_normalization_2_moving_mean:Z(
batch_normalization_2_beta:Z
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs%batch_normalization_2_moving_variancebatch_normalization_2_gamma!batch_normalization_2_moving_meanbatch_normalization_2_beta*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������Z*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1275o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������Z`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*.
_input_shapes
:���������Z: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������Z
 
_user_specified_nameinputs
�
�
__inference_loss_fn_1_3645[
Aconv2d_1_kernel_regularizer_square_readvariableop_conv2d_1_kernel:
identity��1conv2d_1/kernel/Regularizer/Square/ReadVariableOp�
1conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpAconv2d_1_kernel_regularizer_square_readvariableop_conv2d_1_kernel*&
_output_shapes
:*
dtype0�
"conv2d_1/kernel/Regularizer/SquareSquare9conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
!conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_1/kernel/Regularizer/SumSum&conv2d_1/kernel/Regularizer/Square:y:0*conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
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
identityIdentity:output:0*
_input_shapes
: 2f
1conv2d_1/kernel/Regularizer/Square/ReadVariableOp1conv2d_1/kernel/Regularizer/Square/ReadVariableOp
�
�
B__inference_conv2d_1_layer_call_and_return_conditional_losses_1484

inputs?
%conv2d_readvariableop_conv2d_1_kernel:2
$biasadd_readvariableop_conv2d_1_bias:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�1conv2d_1/kernel/Regularizer/Square/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOp%conv2d_readvariableop_conv2d_1_kernel*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������	*
data_formatNCHW*
paddingVALID*
strides
w
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_conv2d_1_bias*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������	*
data_formatNCHW�
1conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%conv2d_readvariableop_conv2d_1_kernel*&
_output_shapes
:*
dtype0�
"conv2d_1/kernel/Regularizer/SquareSquare9conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
!conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_1/kernel/Regularizer/SumSum&conv2d_1/kernel/Regularizer/Square:y:0*conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
conv2d_1/kernel/Regularizer/mulMul*conv2d_1/kernel/Regularizer/mul/x:output:0(conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:���������	�
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp2^conv2d_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2f
1conv2d_1/kernel/Regularizer/Square/ReadVariableOp1conv2d_1/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
'__inference_conv2d_2_layer_call_fn_3355

inputs)
conv2d_2_kernel:
conv2d_2_bias:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_2_kernelconv2d_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_1546w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*2
_input_shapes!
:���������	: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
H
,__inference_max_pooling2d_layer_call_fn_3126

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_max_pooling2d_layer_call_and_return_conditional_losses_987�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
M__inference_batch_normalization_layer_call_and_return_conditional_losses_1019

inputsK
<batchnorm_readvariableop_batch_normalization_moving_variance:	�E
6batchnorm_mul_readvariableop_batch_normalization_gamma:	�I
:batchnorm_readvariableop_1_batch_normalization_moving_mean:	�B
3batchnorm_readvariableop_2_batch_normalization_beta:	�
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp�
batchnorm/ReadVariableOpReadVariableOp<batchnorm_readvariableop_batch_normalization_moving_variance*
_output_shapes	
:�*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:��
batchnorm/mul/ReadVariableOpReadVariableOp6batchnorm_mul_readvariableop_batch_normalization_gamma*
_output_shapes	
:�*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
batchnorm/ReadVariableOp_1ReadVariableOp:batchnorm_readvariableop_1_batch_normalization_moving_mean*
_output_shapes	
:�*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:��
batchnorm/ReadVariableOp_2ReadVariableOp3batchnorm_readvariableop_2_batch_normalization_beta*
_output_shapes	
:�*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
"__inference_signature_wrapper_2612
input_1/
layer_normalization_gamma:x'.
layer_normalization_beta:x''
conv2d_kernel:		x
conv2d_bias:)
conv2d_1_kernel:
conv2d_1_bias:2
#batch_normalization_moving_variance:	�(
batch_normalization_gamma:	�.
batch_normalization_moving_mean:	�'
batch_normalization_beta:	�)
conv2d_2_kernel:
conv2d_2_bias:3
%batch_normalization_1_moving_variance:Z)
batch_normalization_1_gamma:Z/
!batch_normalization_1_moving_mean:Z(
batch_normalization_1_beta:Z3
%batch_normalization_2_moving_variance:Z)
batch_normalization_2_gamma:Z/
!batch_normalization_2_moving_mean:Z(
batch_normalization_2_beta:Z
dense_kernel:Z%

dense_bias:%
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1layer_normalization_gammalayer_normalization_betaconv2d_kernelconv2d_biasconv2d_1_kernelconv2d_1_bias#batch_normalization_moving_variancebatch_normalization_gammabatch_normalization_moving_meanbatch_normalization_betaconv2d_2_kernelconv2d_2_bias%batch_normalization_1_moving_variancebatch_normalization_1_gamma!batch_normalization_1_moving_meanbatch_normalization_1_beta%batch_normalization_2_moving_variancebatch_normalization_2_gamma!batch_normalization_2_moving_meanbatch_normalization_2_betadense_kernel
dense_bias*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������%*8
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *'
f"R 
__inference__wrapped_model_978o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������%`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*Z
_input_shapesI
G:���������x': : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������x'
!
_user_specified_name	input_1
��
�
E__inference_CNN_2D_NORM_layer_call_and_return_conditional_losses_3059

inputsZ
Dlayer_normalization_reshape_readvariableop_layer_normalization_gamma:x'[
Elayer_normalization_reshape_1_readvariableop_layer_normalization_beta:x'D
*conv2d_conv2d_readvariableop_conv2d_kernel:		x7
)conv2d_biasadd_readvariableop_conv2d_bias:H
.conv2d_1_conv2d_readvariableop_conv2d_1_kernel:;
-conv2d_1_biasadd_readvariableop_conv2d_1_bias:a
Rbatch_normalization_assignmovingavg_readvariableop_batch_normalization_moving_mean:	�g
Xbatch_normalization_assignmovingavg_1_readvariableop_batch_normalization_moving_variance:	�Y
Jbatch_normalization_batchnorm_mul_readvariableop_batch_normalization_gamma:	�T
Ebatch_normalization_batchnorm_readvariableop_batch_normalization_beta:	�H
.conv2d_2_conv2d_readvariableop_conv2d_2_kernel:;
-conv2d_2_biasadd_readvariableop_conv2d_2_bias:d
Vbatch_normalization_1_assignmovingavg_readvariableop_batch_normalization_1_moving_mean:Zj
\batch_normalization_1_assignmovingavg_1_readvariableop_batch_normalization_1_moving_variance:Z\
Nbatch_normalization_1_batchnorm_mul_readvariableop_batch_normalization_1_gamma:ZW
Ibatch_normalization_1_batchnorm_readvariableop_batch_normalization_1_beta:Zd
Vbatch_normalization_2_assignmovingavg_readvariableop_batch_normalization_2_moving_mean:Zj
\batch_normalization_2_assignmovingavg_1_readvariableop_batch_normalization_2_moving_variance:Z\
Nbatch_normalization_2_batchnorm_mul_readvariableop_batch_normalization_2_gamma:ZW
Ibatch_normalization_2_batchnorm_readvariableop_batch_normalization_2_beta:Z:
(dense_matmul_readvariableop_dense_kernel:Z%5
'dense_biasadd_readvariableop_dense_bias:%
identity��#batch_normalization/AssignMovingAvg�2batch_normalization/AssignMovingAvg/ReadVariableOp�%batch_normalization/AssignMovingAvg_1�4batch_normalization/AssignMovingAvg_1/ReadVariableOp�,batch_normalization/batchnorm/ReadVariableOp�0batch_normalization/batchnorm/mul/ReadVariableOp�%batch_normalization_1/AssignMovingAvg�4batch_normalization_1/AssignMovingAvg/ReadVariableOp�'batch_normalization_1/AssignMovingAvg_1�6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp�.batch_normalization_1/batchnorm/ReadVariableOp�2batch_normalization_1/batchnorm/mul/ReadVariableOp�%batch_normalization_2/AssignMovingAvg�4batch_normalization_2/AssignMovingAvg/ReadVariableOp�'batch_normalization_2/AssignMovingAvg_1�6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp�.batch_normalization_2/batchnorm/ReadVariableOp�2batch_normalization_2/batchnorm/mul/ReadVariableOp�conv2d/BiasAdd/ReadVariableOp�conv2d/Conv2D/ReadVariableOp�/conv2d/kernel/Regularizer/Square/ReadVariableOp�conv2d_1/BiasAdd/ReadVariableOp�conv2d_1/Conv2D/ReadVariableOp�1conv2d_1/kernel/Regularizer/Square/ReadVariableOp�conv2d_2/BiasAdd/ReadVariableOp�conv2d_2/Conv2D/ReadVariableOp�1conv2d_2/kernel/Regularizer/Square/ReadVariableOp�dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�.dense/kernel/Regularizer/Square/ReadVariableOp�*layer_normalization/Reshape/ReadVariableOp�,layer_normalization/Reshape_1/ReadVariableOp�
2layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         �
 layer_normalization/moments/meanMeaninputs;layer_normalization/moments/mean/reduction_indices:output:0*
T0*/
_output_shapes
:���������*
	keep_dims(�
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*/
_output_shapes
:����������
-layer_normalization/moments/SquaredDifferenceSquaredDifferenceinputs1layer_normalization/moments/StopGradient:output:0*
T0*/
_output_shapes
:���������x'�
6layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         �
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
T0*/
_output_shapes
:���������*
	keep_dims(�
*layer_normalization/Reshape/ReadVariableOpReadVariableOpDlayer_normalization_reshape_readvariableop_layer_normalization_gamma*"
_output_shapes
:x'*
dtype0z
!layer_normalization/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   x      '   �
layer_normalization/ReshapeReshape2layer_normalization/Reshape/ReadVariableOp:value:0*layer_normalization/Reshape/shape:output:0*
T0*&
_output_shapes
:x'�
,layer_normalization/Reshape_1/ReadVariableOpReadVariableOpElayer_normalization_reshape_1_readvariableop_layer_normalization_beta*"
_output_shapes
:x'*
dtype0|
#layer_normalization/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   x      '   �
layer_normalization/Reshape_1Reshape4layer_normalization/Reshape_1/ReadVariableOp:value:0,layer_normalization/Reshape_1/shape:output:0*
T0*&
_output_shapes
:x'h
#layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*/
_output_shapes
:����������
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*/
_output_shapes
:����������
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:0$layer_normalization/Reshape:output:0*
T0*/
_output_shapes
:���������x'�
#layer_normalization/batchnorm/mul_1Mulinputs%layer_normalization/batchnorm/mul:z:0*
T0*/
_output_shapes
:���������x'�
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*/
_output_shapes
:���������x'�
!layer_normalization/batchnorm/subSub&layer_normalization/Reshape_1:output:0'layer_normalization/batchnorm/mul_2:z:0*
T0*/
_output_shapes
:���������x'�
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*/
_output_shapes
:���������x'�
conv2d/Conv2D/ReadVariableOpReadVariableOp*conv2d_conv2d_readvariableop_conv2d_kernel*&
_output_shapes
:		x*
dtype0�
conv2d/Conv2DConv2D'layer_normalization/batchnorm/add_1:z:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
data_formatNCHW*
paddingVALID*
strides
�
conv2d/BiasAdd/ReadVariableOpReadVariableOp)conv2d_biasadd_readvariableop_conv2d_bias*
_output_shapes
:*
dtype0�
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
data_formatNCHW�
max_pooling2d/MaxPoolMaxPoolconv2d/BiasAdd:output:0*/
_output_shapes
:���������*
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
 *���=�
1gaussian_noise/random_normal/RandomStandardNormalRandomStandardNormalgaussian_noise/Shape:output:0*
T0*/
_output_shapes
:���������*
dtype0�
 gaussian_noise/random_normal/mulMul:gaussian_noise/random_normal/RandomStandardNormal:output:0,gaussian_noise/random_normal/stddev:output:0*
T0*/
_output_shapes
:����������
gaussian_noise/random_normalAddV2$gaussian_noise/random_normal/mul:z:0*gaussian_noise/random_normal/mean:output:0*
T0*/
_output_shapes
:����������
gaussian_noise/addAddV2max_pooling2d/MaxPool:output:0 gaussian_noise/random_normal:z:0*
T0*/
_output_shapes
:���������i
activation/ReluRelugaussian_noise/add:z:0*
T0*/
_output_shapes
:����������
conv2d_1/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_conv2d_readvariableop_conv2d_1_kernel*&
_output_shapes
:*
dtype0�
conv2d_1/Conv2DConv2Dactivation/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������	*
data_formatNCHW*
paddingVALID*
strides
�
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp-conv2d_1_biasadd_readvariableop_conv2d_1_bias*
_output_shapes
:*
dtype0�
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������	*
data_formatNCHW^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  �
flatten/ReshapeReshapeconv2d_1/BiasAdd:output:0flatten/Const:output:0*
T0*(
_output_shapes
:����������|
2batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
 batch_normalization/moments/meanMeanflatten/Reshape:output:0;batch_normalization/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
(batch_normalization/moments/StopGradientStopGradient)batch_normalization/moments/mean:output:0*
T0*
_output_shapes
:	��
-batch_normalization/moments/SquaredDifferenceSquaredDifferenceflatten/Reshape:output:01batch_normalization/moments/StopGradient:output:0*
T0*(
_output_shapes
:�����������
6batch_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
$batch_normalization/moments/varianceMean1batch_normalization/moments/SquaredDifference:z:0?batch_normalization/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
#batch_normalization/moments/SqueezeSqueeze)batch_normalization/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 �
%batch_normalization/moments/Squeeze_1Squeeze-batch_normalization/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 n
)batch_normalization/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
2batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOpRbatch_normalization_assignmovingavg_readvariableop_batch_normalization_moving_mean*
_output_shapes	
:�*
dtype0�
'batch_normalization/AssignMovingAvg/subSub:batch_normalization/AssignMovingAvg/ReadVariableOp:value:0,batch_normalization/moments/Squeeze:output:0*
T0*
_output_shapes	
:��
'batch_normalization/AssignMovingAvg/mulMul+batch_normalization/AssignMovingAvg/sub:z:02batch_normalization/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
#batch_normalization/AssignMovingAvgAssignSubVariableOpRbatch_normalization_assignmovingavg_readvariableop_batch_normalization_moving_mean+batch_normalization/AssignMovingAvg/mul:z:03^batch_normalization/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0p
+batch_normalization/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
4batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOpXbatch_normalization_assignmovingavg_1_readvariableop_batch_normalization_moving_variance*
_output_shapes	
:�*
dtype0�
)batch_normalization/AssignMovingAvg_1/subSub<batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:0.batch_normalization/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:��
)batch_normalization/AssignMovingAvg_1/mulMul-batch_normalization/AssignMovingAvg_1/sub:z:04batch_normalization/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
%batch_normalization/AssignMovingAvg_1AssignSubVariableOpXbatch_normalization_assignmovingavg_1_readvariableop_batch_normalization_moving_variance-batch_normalization/AssignMovingAvg_1/mul:z:05^batch_normalization/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0h
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
!batch_normalization/batchnorm/addAddV2.batch_normalization/moments/Squeeze_1:output:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�y
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:��
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOpJbatch_normalization_batchnorm_mul_readvariableop_batch_normalization_gamma*
_output_shapes	
:�*
dtype0�
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
#batch_normalization/batchnorm/mul_1Mulflatten/Reshape:output:0%batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
#batch_normalization/batchnorm/mul_2Mul,batch_normalization/moments/Squeeze:output:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
,batch_normalization/batchnorm/ReadVariableOpReadVariableOpEbatch_normalization_batchnorm_readvariableop_batch_normalization_beta*
_output_shapes	
:�*
dtype0�
!batch_normalization/batchnorm/subSub4batch_normalization/batchnorm/ReadVariableOp:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������d
reshape/ShapeShape'batch_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:e
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
valueB:�
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Y
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Y
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :	�
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
reshape/ReshapeReshape'batch_normalization/batchnorm/add_1:z:0reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:���������	^
gaussian_noise_1/ShapeShapereshape/Reshape:output:0*
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
 *���=�
3gaussian_noise_1/random_normal/RandomStandardNormalRandomStandardNormalgaussian_noise_1/Shape:output:0*
T0*/
_output_shapes
:���������	*
dtype0�
"gaussian_noise_1/random_normal/mulMul<gaussian_noise_1/random_normal/RandomStandardNormal:output:0.gaussian_noise_1/random_normal/stddev:output:0*
T0*/
_output_shapes
:���������	�
gaussian_noise_1/random_normalAddV2&gaussian_noise_1/random_normal/mul:z:0,gaussian_noise_1/random_normal/mean:output:0*
T0*/
_output_shapes
:���������	�
gaussian_noise_1/addAddV2reshape/Reshape:output:0"gaussian_noise_1/random_normal:z:0*
T0*/
_output_shapes
:���������	m
activation_1/ReluRelugaussian_noise_1/add:z:0*
T0*/
_output_shapes
:���������	�
conv2d_2/Conv2D/ReadVariableOpReadVariableOp.conv2d_2_conv2d_readvariableop_conv2d_2_kernel*&
_output_shapes
:*
dtype0�
conv2d_2/Conv2DConv2Dactivation_1/Relu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
data_formatNCHW*
paddingVALID*
strides
�
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp-conv2d_2_biasadd_readvariableop_conv2d_2_bias*
_output_shapes
:*
dtype0�
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
data_formatNCHW`
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
flatten_1/ReshapeReshapeconv2d_2/BiasAdd:output:0flatten_1/Const:output:0*
T0*'
_output_shapes
:���������Z~
4batch_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
"batch_normalization_1/moments/meanMeanflatten_1/Reshape:output:0=batch_normalization_1/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:Z*
	keep_dims(�
*batch_normalization_1/moments/StopGradientStopGradient+batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes

:Z�
/batch_normalization_1/moments/SquaredDifferenceSquaredDifferenceflatten_1/Reshape:output:03batch_normalization_1/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������Z�
8batch_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
&batch_normalization_1/moments/varianceMean3batch_normalization_1/moments/SquaredDifference:z:0Abatch_normalization_1/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:Z*
	keep_dims(�
%batch_normalization_1/moments/SqueezeSqueeze+batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes
:Z*
squeeze_dims
 �
'batch_normalization_1/moments/Squeeze_1Squeeze/batch_normalization_1/moments/variance:output:0*
T0*
_output_shapes
:Z*
squeeze_dims
 p
+batch_normalization_1/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
4batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOpVbatch_normalization_1_assignmovingavg_readvariableop_batch_normalization_1_moving_mean*
_output_shapes
:Z*
dtype0�
)batch_normalization_1/AssignMovingAvg/subSub<batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_1/moments/Squeeze:output:0*
T0*
_output_shapes
:Z�
)batch_normalization_1/AssignMovingAvg/mulMul-batch_normalization_1/AssignMovingAvg/sub:z:04batch_normalization_1/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:Z�
%batch_normalization_1/AssignMovingAvgAssignSubVariableOpVbatch_normalization_1_assignmovingavg_readvariableop_batch_normalization_1_moving_mean-batch_normalization_1/AssignMovingAvg/mul:z:05^batch_normalization_1/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0r
-batch_normalization_1/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOp\batch_normalization_1_assignmovingavg_1_readvariableop_batch_normalization_1_moving_variance*
_output_shapes
:Z*
dtype0�
+batch_normalization_1/AssignMovingAvg_1/subSub>batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_1/moments/Squeeze_1:output:0*
T0*
_output_shapes
:Z�
+batch_normalization_1/AssignMovingAvg_1/mulMul/batch_normalization_1/AssignMovingAvg_1/sub:z:06batch_normalization_1/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:Z�
'batch_normalization_1/AssignMovingAvg_1AssignSubVariableOp\batch_normalization_1_assignmovingavg_1_readvariableop_batch_normalization_1_moving_variance/batch_normalization_1/AssignMovingAvg_1/mul:z:07^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0j
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
#batch_normalization_1/batchnorm/addAddV20batch_normalization_1/moments/Squeeze_1:output:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes
:Z|
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
:Z�
2batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpNbatch_normalization_1_batchnorm_mul_readvariableop_batch_normalization_1_gamma*
_output_shapes
:Z*
dtype0�
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:0:batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Z�
%batch_normalization_1/batchnorm/mul_1Mulflatten_1/Reshape:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������Z�
%batch_normalization_1/batchnorm/mul_2Mul.batch_normalization_1/moments/Squeeze:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
:Z�
.batch_normalization_1/batchnorm/ReadVariableOpReadVariableOpIbatch_normalization_1_batchnorm_readvariableop_batch_normalization_1_beta*
_output_shapes
:Z*
dtype0�
#batch_normalization_1/batchnorm/subSub6batch_normalization_1/batchnorm/ReadVariableOp:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Z�
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������Zh
reshape_1/ShapeShape)batch_normalization_1/batchnorm/add_1:z:0*
T0*
_output_shapes
:g
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
valueB:�
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0"reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
reshape_1/ReshapeReshape)batch_normalization_1/batchnorm/add_1:z:0 reshape_1/Reshape/shape:output:0*
T0*/
_output_shapes
:���������`
gaussian_noise_2/ShapeShapereshape_1/Reshape:output:0*
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
 *���=�
3gaussian_noise_2/random_normal/RandomStandardNormalRandomStandardNormalgaussian_noise_2/Shape:output:0*
T0*/
_output_shapes
:���������*
dtype0�
"gaussian_noise_2/random_normal/mulMul<gaussian_noise_2/random_normal/RandomStandardNormal:output:0.gaussian_noise_2/random_normal/stddev:output:0*
T0*/
_output_shapes
:����������
gaussian_noise_2/random_normalAddV2&gaussian_noise_2/random_normal/mul:z:0,gaussian_noise_2/random_normal/mean:output:0*
T0*/
_output_shapes
:����������
gaussian_noise_2/addAddV2reshape_1/Reshape:output:0"gaussian_noise_2/random_normal:z:0*
T0*/
_output_shapes
:���������m
activation_2/ReluRelugaussian_noise_2/add:z:0*
T0*/
_output_shapes
:���������`
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
flatten_2/ReshapeReshapeactivation_2/Relu:activations:0flatten_2/Const:output:0*
T0*'
_output_shapes
:���������Z~
4batch_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
"batch_normalization_2/moments/meanMeanflatten_2/Reshape:output:0=batch_normalization_2/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:Z*
	keep_dims(�
*batch_normalization_2/moments/StopGradientStopGradient+batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes

:Z�
/batch_normalization_2/moments/SquaredDifferenceSquaredDifferenceflatten_2/Reshape:output:03batch_normalization_2/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������Z�
8batch_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
&batch_normalization_2/moments/varianceMean3batch_normalization_2/moments/SquaredDifference:z:0Abatch_normalization_2/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:Z*
	keep_dims(�
%batch_normalization_2/moments/SqueezeSqueeze+batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes
:Z*
squeeze_dims
 �
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
�#<�
4batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOpVbatch_normalization_2_assignmovingavg_readvariableop_batch_normalization_2_moving_mean*
_output_shapes
:Z*
dtype0�
)batch_normalization_2/AssignMovingAvg/subSub<batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_2/moments/Squeeze:output:0*
T0*
_output_shapes
:Z�
)batch_normalization_2/AssignMovingAvg/mulMul-batch_normalization_2/AssignMovingAvg/sub:z:04batch_normalization_2/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:Z�
%batch_normalization_2/AssignMovingAvgAssignSubVariableOpVbatch_normalization_2_assignmovingavg_readvariableop_batch_normalization_2_moving_mean-batch_normalization_2/AssignMovingAvg/mul:z:05^batch_normalization_2/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0r
-batch_normalization_2/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOp\batch_normalization_2_assignmovingavg_1_readvariableop_batch_normalization_2_moving_variance*
_output_shapes
:Z*
dtype0�
+batch_normalization_2/AssignMovingAvg_1/subSub>batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_2/moments/Squeeze_1:output:0*
T0*
_output_shapes
:Z�
+batch_normalization_2/AssignMovingAvg_1/mulMul/batch_normalization_2/AssignMovingAvg_1/sub:z:06batch_normalization_2/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:Z�
'batch_normalization_2/AssignMovingAvg_1AssignSubVariableOp\batch_normalization_2_assignmovingavg_1_readvariableop_batch_normalization_2_moving_variance/batch_normalization_2/AssignMovingAvg_1/mul:z:07^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0j
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
#batch_normalization_2/batchnorm/addAddV20batch_normalization_2/moments/Squeeze_1:output:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:Z|
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:Z�
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpNbatch_normalization_2_batchnorm_mul_readvariableop_batch_normalization_2_gamma*
_output_shapes
:Z*
dtype0�
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Z�
%batch_normalization_2/batchnorm/mul_1Mulflatten_2/Reshape:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������Z�
%batch_normalization_2/batchnorm/mul_2Mul.batch_normalization_2/moments/Squeeze:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:Z�
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOpIbatch_normalization_2_batchnorm_readvariableop_batch_normalization_2_beta*
_output_shapes
:Z*
dtype0�
#batch_normalization_2/batchnorm/subSub6batch_normalization_2/batchnorm/ReadVariableOp:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Z�
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������Z�
dense/MatMul/ReadVariableOpReadVariableOp(dense_matmul_readvariableop_dense_kernel*
_output_shapes

:Z%*
dtype0�
dense/MatMulMatMul)batch_normalization_2/batchnorm/add_1:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������%�
dense/BiasAdd/ReadVariableOpReadVariableOp'dense_biasadd_readvariableop_dense_bias*
_output_shapes
:%*
dtype0�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������%n
dense/ActivityRegularizer/AbsAbsdense/BiasAdd:output:0*
T0*'
_output_shapes
:���������%p
dense/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense/ActivityRegularizer/SumSum!dense/ActivityRegularizer/Abs:y:0(dense/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: d
dense/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
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
valueB:�
'dense/ActivityRegularizer/strided_sliceStridedSlice(dense/ActivityRegularizer/Shape:output:06dense/ActivityRegularizer/strided_slice/stack:output:08dense/ActivityRegularizer/strided_slice/stack_1:output:08dense/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
dense/ActivityRegularizer/CastCast0dense/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: �
!dense/ActivityRegularizer/truedivRealDiv!dense/ActivityRegularizer/mul:z:0"dense/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: k
activation_3/SoftplusSoftplusdense/BiasAdd:output:0*
T0*'
_output_shapes
:���������%�
/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp*conv2d_conv2d_readvariableop_conv2d_kernel*&
_output_shapes
:		x*
dtype0�
 conv2d/kernel/Regularizer/SquareSquare7conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		xx
conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d/kernel/Regularizer/SumSum$conv2d/kernel/Regularizer/Square:y:0(conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
conv2d/kernel/Regularizer/mulMul(conv2d/kernel/Regularizer/mul/x:output:0&conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp.conv2d_1_conv2d_readvariableop_conv2d_1_kernel*&
_output_shapes
:*
dtype0�
"conv2d_1/kernel/Regularizer/SquareSquare9conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
!conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_1/kernel/Regularizer/SumSum&conv2d_1/kernel/Regularizer/Square:y:0*conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
conv2d_1/kernel/Regularizer/mulMul*conv2d_1/kernel/Regularizer/mul/x:output:0(conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp.conv2d_2_conv2d_readvariableop_conv2d_2_kernel*&
_output_shapes
:*
dtype0�
"conv2d_2/kernel/Regularizer/SquareSquare9conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
!conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_2/kernel/Regularizer/SumSum&conv2d_2/kernel/Regularizer/Square:y:0*conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
conv2d_2/kernel/Regularizer/mulMul*conv2d_2/kernel/Regularizer/mul/x:output:0(conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(dense_matmul_readvariableop_dense_kernel*
_output_shapes

:Z%*
dtype0�
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:Z%o
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: r
IdentityIdentity#activation_3/Softplus:activations:0^NoOp*
T0*'
_output_shapes
:���������%�
NoOpNoOp$^batch_normalization/AssignMovingAvg3^batch_normalization/AssignMovingAvg/ReadVariableOp&^batch_normalization/AssignMovingAvg_15^batch_normalization/AssignMovingAvg_1/ReadVariableOp-^batch_normalization/batchnorm/ReadVariableOp1^batch_normalization/batchnorm/mul/ReadVariableOp&^batch_normalization_1/AssignMovingAvg5^batch_normalization_1/AssignMovingAvg/ReadVariableOp(^batch_normalization_1/AssignMovingAvg_17^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp3^batch_normalization_1/batchnorm/mul/ReadVariableOp&^batch_normalization_2/AssignMovingAvg5^batch_normalization_2/AssignMovingAvg/ReadVariableOp(^batch_normalization_2/AssignMovingAvg_17^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_2/batchnorm/ReadVariableOp3^batch_normalization_2/batchnorm/mul/ReadVariableOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp0^conv2d/kernel/Regularizer/Square/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp2^conv2d_1/kernel/Regularizer/Square/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp2^conv2d_2/kernel/Regularizer/Square/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp+^layer_normalization/Reshape/ReadVariableOp-^layer_normalization/Reshape_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*Z
_input_shapesI
G:���������x': : : : : : : : : : : : : : : : : : : : : : 2J
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
2batch_normalization_2/batchnorm/mul/ReadVariableOp2batch_normalization_2/batchnorm/mul/ReadVariableOp2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2b
/conv2d/kernel/Regularizer/Square/ReadVariableOp/conv2d/kernel/Regularizer/Square/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2f
1conv2d_1/kernel/Regularizer/Square/ReadVariableOp1conv2d_1/kernel/Regularizer/Square/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2f
1conv2d_2/kernel/Regularizer/Square/ReadVariableOp1conv2d_2/kernel/Regularizer/Square/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2X
*layer_normalization/Reshape/ReadVariableOp*layer_normalization/Reshape/ReadVariableOp2\
,layer_normalization/Reshape_1/ReadVariableOp,layer_normalization/Reshape_1/ReadVariableOp:W S
/
_output_shapes
:���������x'
 
_user_specified_nameinputs
�
b
F__inference_max_pooling2d_layer_call_and_return_conditional_losses_987

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
data_formatNCHW*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�&
�
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3591

inputsN
@assignmovingavg_readvariableop_batch_normalization_2_moving_mean:ZT
Fassignmovingavg_1_readvariableop_batch_normalization_2_moving_variance:ZF
8batchnorm_mul_readvariableop_batch_normalization_2_gamma:ZA
3batchnorm_readvariableop_batch_normalization_2_beta:Z
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
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

:Z�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������Zl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
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
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp@assignmovingavg_readvariableop_batch_normalization_2_moving_mean*
_output_shapes
:Z*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:Zx
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:Z�
AssignMovingAvgAssignSubVariableOp@assignmovingavg_readvariableop_batch_normalization_2_moving_meanAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpFassignmovingavg_1_readvariableop_batch_normalization_2_moving_variance*
_output_shapes
:Z*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:Z~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:Z�
AssignMovingAvg_1AssignSubVariableOpFassignmovingavg_1_readvariableop_batch_normalization_2_moving_varianceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:ZP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:Z�
batchnorm/mul/ReadVariableOpReadVariableOp8batchnorm_mul_readvariableop_batch_normalization_2_gamma*
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
:���������Zh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:Z�
batchnorm/ReadVariableOpReadVariableOp3batchnorm_readvariableop_batch_normalization_2_beta*
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
:���������Zb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������Z�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*.
_input_shapes
:���������Z: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������Z
 
_user_specified_nameinputs
�
�
%__inference_conv2d_layer_call_fn_3105

inputs'
conv2d_kernel:		x
conv2d_bias:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_kernelconv2d_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_1445w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*2
_input_shapes!
:���������x': : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������x'
 
_user_specified_nameinputs
�
d
H__inference_gaussian_noise_layer_call_and_return_conditional_losses_3155

inputs
identityV
IdentityIdentityinputs*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
]
A__inference_reshape_layer_call_and_return_conditional_losses_3307

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
valueB:�
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
value	B :	�
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:���������	`
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:���������	"
identityIdentity:output:0*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
2__inference_batch_normalization_layer_call_fn_3234

inputs.
batch_normalization_moving_mean:	�2
#batch_normalization_moving_variance:	�(
batch_normalization_gamma:	�'
batch_normalization_beta:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_moving_mean#batch_normalization_moving_variancebatch_normalization_gammabatch_normalization_beta*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_1062p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*/
_input_shapes
:����������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
4__inference_batch_normalization_1_layer_call_fn_3400

inputs/
!batch_normalization_1_moving_mean:Z3
%batch_normalization_1_moving_variance:Z)
batch_normalization_1_gamma:Z(
batch_normalization_1_beta:Z
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs!batch_normalization_1_moving_mean%batch_normalization_1_moving_variancebatch_normalization_1_gammabatch_normalization_1_beta*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������Z*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1190o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������Z`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*.
_input_shapes
:���������Z: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������Z
 
_user_specified_nameinputs
�p
�
 __inference__traced_restore_3884
file_prefix@
*assignvariableop_layer_normalization_gamma:x'A
+assignvariableop_1_layer_normalization_beta:x':
 assignvariableop_2_conv2d_kernel:		x,
assignvariableop_3_conv2d_bias:<
"assignvariableop_4_conv2d_1_kernel:.
 assignvariableop_5_conv2d_1_bias:;
,assignvariableop_6_batch_normalization_gamma:	�:
+assignvariableop_7_batch_normalization_beta:	�A
2assignvariableop_8_batch_normalization_moving_mean:	�E
6assignvariableop_9_batch_normalization_moving_variance:	�=
#assignvariableop_10_conv2d_2_kernel:/
!assignvariableop_11_conv2d_2_bias:=
/assignvariableop_12_batch_normalization_1_gamma:Z<
.assignvariableop_13_batch_normalization_1_beta:ZC
5assignvariableop_14_batch_normalization_1_moving_mean:ZG
9assignvariableop_15_batch_normalization_1_moving_variance:Z=
/assignvariableop_16_batch_normalization_2_gamma:Z<
.assignvariableop_17_batch_normalization_2_beta:ZC
5assignvariableop_18_batch_normalization_2_moving_mean:ZG
9assignvariableop_19_batch_normalization_2_moving_variance:Z2
 assignvariableop_20_dense_kernel:Z%,
assignvariableop_21_dense_bias:%#
assignvariableop_22_total: #
assignvariableop_23_count: %
assignvariableop_24_total_1: %
assignvariableop_25_count_1: %
assignvariableop_26_total_2: %
assignvariableop_27_count_2: 
identity_29��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*M
valueDBBB B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapesv
t:::::::::::::::::::::::::::::*+
dtypes!
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp*assignvariableop_layer_normalization_gammaIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp+assignvariableop_1_layer_normalization_betaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp assignvariableop_2_conv2d_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpassignvariableop_3_conv2d_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv2d_1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv2d_1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp,assignvariableop_6_batch_normalization_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp+assignvariableop_7_batch_normalization_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp2assignvariableop_8_batch_normalization_moving_meanIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp6assignvariableop_9_batch_normalization_moving_varianceIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp#assignvariableop_10_conv2d_2_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp!assignvariableop_11_conv2d_2_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp/assignvariableop_12_batch_normalization_1_gammaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp.assignvariableop_13_batch_normalization_1_betaIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp5assignvariableop_14_batch_normalization_1_moving_meanIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp9assignvariableop_15_batch_normalization_1_moving_varianceIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp/assignvariableop_16_batch_normalization_2_gammaIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp.assignvariableop_17_batch_normalization_2_betaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp5assignvariableop_18_batch_normalization_2_moving_meanIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp9assignvariableop_19_batch_normalization_2_moving_varianceIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp assignvariableop_20_dense_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOpassignvariableop_21_dense_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOpassignvariableop_22_totalIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOpassignvariableop_23_countIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOpassignvariableop_24_total_1Identity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOpassignvariableop_25_count_1Identity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOpassignvariableop_26_total_2Identity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOpassignvariableop_27_count_2Identity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_28Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_29IdentityIdentity_28:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_29Identity_29:output:0*M
_input_shapes<
:: : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_27AssignVariableOp_272(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
E
)__inference_activation_layer_call_fn_3171

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_activation_layer_call_and_return_conditional_losses_1466h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
B
+__inference_dense_activity_regularizer_1636
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
:���������D
SumSumAbs:y:0range:output:0*
T0*
_output_shapes
: J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:I
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
�
�
?__inference_dense_layer_call_and_return_conditional_losses_1621

inputs4
"matmul_readvariableop_dense_kernel:Z%/
!biasadd_readvariableop_dense_bias:%
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�.dense/kernel/Regularizer/Square/ReadVariableOpx
MatMul/ReadVariableOpReadVariableOp"matmul_readvariableop_dense_kernel*
_output_shapes

:Z%*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������%t
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_dense_bias*
_output_shapes
:%*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������%�
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp"matmul_readvariableop_dense_kernel*
_output_shapes

:Z%*
dtype0�
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:Z%o
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������%�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������Z: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������Z
 
_user_specified_nameinputs
�	
i
J__inference_gaussian_noise_1_layer_call_and_return_conditional_losses_1893

inputs
identity�;
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
 *���=�
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*/
_output_shapes
:���������	*
dtype0�
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*/
_output_shapes
:���������	�
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*/
_output_shapes
:���������	a
addAddV2inputsrandom_normal:z:0*
T0*/
_output_shapes
:���������	W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:���������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������	:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�=
�
__inference__traced_save_3790
file_prefix8
4savev2_layer_normalization_gamma_read_readvariableop7
3savev2_layer_normalization_beta_read_readvariableop,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop:
6savev2_batch_normalization_1_gamma_read_readvariableop9
5savev2_batch_normalization_1_beta_read_readvariableop@
<savev2_batch_normalization_1_moving_mean_read_readvariableopD
@savev2_batch_normalization_1_moving_variance_read_readvariableop:
6savev2_batch_normalization_2_gamma_read_readvariableop9
5savev2_batch_normalization_2_beta_read_readvariableop@
<savev2_batch_normalization_2_moving_mean_read_readvariableopD
@savev2_batch_normalization_2_moving_variance_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
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
_temp/part�
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
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*M
valueDBBB B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:04savev2_layer_normalization_gamma_read_readvariableop3savev2_layer_normalization_beta_read_readvariableop(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop6savev2_batch_normalization_2_gamma_read_readvariableop5savev2_batch_normalization_2_beta_read_readvariableop<savev2_batch_normalization_2_moving_mean_read_readvariableop@savev2_batch_normalization_2_moving_variance_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *+
dtypes!
2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
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

identity_1Identity_1:output:0*�
_input_shapes�
�: :x':x':		x::::�:�:�:�:::Z:Z:Z:Z:Z:Z:Z:Z:Z%:%: : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:($
"
_output_shapes
:x':($
"
_output_shapes
:x':,(
&
_output_shapes
:		x: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::!

_output_shapes	
:�:!

_output_shapes	
:�:!	

_output_shapes	
:�:!


_output_shapes	
:�:,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:Z: 

_output_shapes
:Z: 

_output_shapes
:Z: 

_output_shapes
:Z: 

_output_shapes
:Z: 

_output_shapes
:Z: 

_output_shapes
:Z: 

_output_shapes
:Z:$ 

_output_shapes

:Z%: 

_output_shapes
:%:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
@__inference_conv2d_layer_call_and_return_conditional_losses_1445

inputs=
#conv2d_readvariableop_conv2d_kernel:		x0
"biasadd_readvariableop_conv2d_bias:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�/conv2d/kernel/Regularizer/Square/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOp#conv2d_readvariableop_conv2d_kernel*&
_output_shapes
:		x*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
data_formatNCHW*
paddingVALID*
strides
u
BiasAdd/ReadVariableOpReadVariableOp"biasadd_readvariableop_conv2d_bias*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
data_formatNCHW�
/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp#conv2d_readvariableop_conv2d_kernel*&
_output_shapes
:		x*
dtype0�
 conv2d/kernel/Regularizer/SquareSquare7conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		xx
conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d/kernel/Regularizer/SumSum$conv2d/kernel/Regularizer/Square:y:0(conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
conv2d/kernel/Regularizer/mulMul(conv2d/kernel/Regularizer/mul/x:output:0&conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp0^conv2d/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������x': : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2b
/conv2d/kernel/Regularizer/Square/ReadVariableOp/conv2d/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������x'
 
_user_specified_nameinputs
�
i
J__inference_gaussian_noise_2_layer_call_and_return_conditional_losses_3498

inputs
identity�;
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
 *���=�
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*/
_output_shapes
:���������*
dtype0�
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*/
_output_shapes
:����������
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*/
_output_shapes
:���������a
addAddV2inputsrandom_normal:z:0*
T0*/
_output_shapes
:���������W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
B__inference_conv2d_2_layer_call_and_return_conditional_losses_3371

inputs?
%conv2d_readvariableop_conv2d_2_kernel:2
$biasadd_readvariableop_conv2d_2_bias:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�1conv2d_2/kernel/Regularizer/Square/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOp%conv2d_readvariableop_conv2d_2_kernel*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
data_formatNCHW*
paddingVALID*
strides
w
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_conv2d_2_bias*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
data_formatNCHW�
1conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%conv2d_readvariableop_conv2d_2_kernel*&
_output_shapes
:*
dtype0�
"conv2d_2/kernel/Regularizer/SquareSquare9conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
!conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_2/kernel/Regularizer/SumSum&conv2d_2/kernel/Regularizer/Square:y:0*conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
conv2d_2/kernel/Regularizer/mulMul*conv2d_2/kernel/Regularizer/mul/x:output:0(conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp2^conv2d_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*2
_input_shapes!
:���������	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2f
1conv2d_2/kernel/Regularizer/Square/ReadVariableOp1conv2d_2/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
�
B__inference_conv2d_2_layer_call_and_return_conditional_losses_1546

inputs?
%conv2d_readvariableop_conv2d_2_kernel:2
$biasadd_readvariableop_conv2d_2_bias:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�1conv2d_2/kernel/Regularizer/Square/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOp%conv2d_readvariableop_conv2d_2_kernel*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
data_formatNCHW*
paddingVALID*
strides
w
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_conv2d_2_bias*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
data_formatNCHW�
1conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%conv2d_readvariableop_conv2d_2_kernel*&
_output_shapes
:*
dtype0�
"conv2d_2/kernel/Regularizer/SquareSquare9conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
!conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_2/kernel/Regularizer/SumSum&conv2d_2/kernel/Regularizer/Square:y:0*conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
conv2d_2/kernel/Regularizer/mulMul*conv2d_2/kernel/Regularizer/mul/x:output:0(conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp2^conv2d_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2f
1conv2d_2/kernel/Regularizer/Square/ReadVariableOp1conv2d_2/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
_
C__inference_flatten_1_layer_call_and_return_conditional_losses_1556

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����Z   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:���������ZX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:���������Z"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
2__inference_layer_normalization_layer_call_fn_3066

inputs/
layer_normalization_gamma:x'.
layer_normalization_beta:x'
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputslayer_normalization_gammalayer_normalization_beta*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������x'*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *V
fQRO
M__inference_layer_normalization_layer_call_and_return_conditional_losses_1425w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������x'`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*2
_input_shapes!
:���������x': : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������x'
 
_user_specified_nameinputs
�&
�
M__inference_batch_normalization_layer_call_and_return_conditional_losses_3288

inputsM
>assignmovingavg_readvariableop_batch_normalization_moving_mean:	�S
Dassignmovingavg_1_readvariableop_batch_normalization_moving_variance:	�E
6batchnorm_mul_readvariableop_batch_normalization_gamma:	�@
1batchnorm_readvariableop_batch_normalization_beta:	�
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	��
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp>assignmovingavg_readvariableop_batch_normalization_moving_mean*
_output_shapes	
:�*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:�y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvgAssignSubVariableOp>assignmovingavg_readvariableop_batch_normalization_moving_meanAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpDassignmovingavg_1_readvariableop_batch_normalization_moving_variance*
_output_shapes	
:�*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvg_1AssignSubVariableOpDassignmovingavg_1_readvariableop_batch_normalization_moving_varianceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:��
batchnorm/mul/ReadVariableOpReadVariableOp6batchnorm_mul_readvariableop_batch_normalization_gamma*
_output_shapes	
:�*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:��
batchnorm/ReadVariableOpReadVariableOp1batchnorm_readvariableop_batch_normalization_beta*
_output_shapes	
:�*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*/
_input_shapes
:����������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
b
F__inference_activation_1_layer_call_and_return_conditional_losses_3342

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:���������	b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������	"
identityIdentity:output:0*.
_input_shapes
:���������	:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
b
F__inference_activation_2_layer_call_and_return_conditional_losses_1590

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:���������b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�&
�
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1318

inputsN
@assignmovingavg_readvariableop_batch_normalization_2_moving_mean:ZT
Fassignmovingavg_1_readvariableop_batch_normalization_2_moving_variance:ZF
8batchnorm_mul_readvariableop_batch_normalization_2_gamma:ZA
3batchnorm_readvariableop_batch_normalization_2_beta:Z
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
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

:Z�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������Zl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
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
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp@assignmovingavg_readvariableop_batch_normalization_2_moving_mean*
_output_shapes
:Z*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:Zx
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:Z�
AssignMovingAvgAssignSubVariableOp@assignmovingavg_readvariableop_batch_normalization_2_moving_meanAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpFassignmovingavg_1_readvariableop_batch_normalization_2_moving_variance*
_output_shapes
:Z*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:Z~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:Z�
AssignMovingAvg_1AssignSubVariableOpFassignmovingavg_1_readvariableop_batch_normalization_2_moving_varianceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:ZP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:Z�
batchnorm/mul/ReadVariableOpReadVariableOp8batchnorm_mul_readvariableop_batch_normalization_2_gamma*
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
:���������Zh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:Z�
batchnorm/ReadVariableOpReadVariableOp3batchnorm_readvariableop_batch_normalization_2_beta*
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
:���������Zb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������Z�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������Z: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������Z
 
_user_specified_nameinputs
�	
�
2__inference_batch_normalization_layer_call_fn_3225

inputs2
#batch_normalization_moving_variance:	�(
batch_normalization_gamma:	�.
batch_normalization_moving_mean:	�'
batch_normalization_beta:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs#batch_normalization_moving_variancebatch_normalization_gammabatch_normalization_moving_meanbatch_normalization_beta*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_1019p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*/
_input_shapes
:����������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3420

inputsL
>batchnorm_readvariableop_batch_normalization_1_moving_variance:ZF
8batchnorm_mul_readvariableop_batch_normalization_1_gamma:ZJ
<batchnorm_readvariableop_1_batch_normalization_1_moving_mean:ZC
5batchnorm_readvariableop_2_batch_normalization_1_beta:Z
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp�
batchnorm/ReadVariableOpReadVariableOp>batchnorm_readvariableop_batch_normalization_1_moving_variance*
_output_shapes
:Z*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:ZP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:Z�
batchnorm/mul/ReadVariableOpReadVariableOp8batchnorm_mul_readvariableop_batch_normalization_1_gamma*
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
:���������Z�
batchnorm/ReadVariableOp_1ReadVariableOp<batchnorm_readvariableop_1_batch_normalization_1_moving_mean*
_output_shapes
:Z*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:Z�
batchnorm/ReadVariableOp_2ReadVariableOp5batchnorm_readvariableop_2_batch_normalization_1_beta*
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
:���������Zb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������Z�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*.
_input_shapes
:���������Z: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������Z
 
_user_specified_nameinputs
�
h
/__inference_gaussian_noise_2_layer_call_fn_3483

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *S
fNRL
J__inference_gaussian_noise_2_layer_call_and_return_conditional_losses_1789w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*.
_input_shapes
:���������22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
c
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_1453

inputs
identity�
MaxPoolMaxPoolinputs*/
_output_shapes
:���������*
data_formatNCHW*
ksize
*
paddingVALID*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
C__inference_dense_layer_call_and_return_all_conditional_losses_3613

inputs
dense_kernel:Z%

dense_bias:%
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsdense_kernel
dense_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������%*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_1621�
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
 *2
config_proto" 

CPU

GPU2 *0J 8� *4
f/R-
+__inference_dense_activity_regularizer_1636o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������%X

Identity_1IdentityPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0**
_input_shapes
:���������Z: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������Z
 
_user_specified_nameinputs
�
�
__inference_loss_fn_2_3656[
Aconv2d_2_kernel_regularizer_square_readvariableop_conv2d_2_kernel:
identity��1conv2d_2/kernel/Regularizer/Square/ReadVariableOp�
1conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpAconv2d_2_kernel_regularizer_square_readvariableop_conv2d_2_kernel*&
_output_shapes
:*
dtype0�
"conv2d_2/kernel/Regularizer/SquareSquare9conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
!conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_2/kernel/Regularizer/SumSum&conv2d_2/kernel/Regularizer/Square:y:0*conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
conv2d_2/kernel/Regularizer/mulMul*conv2d_2/kernel/Regularizer/mul/x:output:0(conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentity#conv2d_2/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: z
NoOpNoOp2^conv2d_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*
_input_shapes
: 2f
1conv2d_2/kernel/Regularizer/Square/ReadVariableOp1conv2d_2/kernel/Regularizer/Square/ReadVariableOp
�
c
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_3136

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
data_formatNCHW*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
@__inference_conv2d_layer_call_and_return_conditional_losses_3121

inputs=
#conv2d_readvariableop_conv2d_kernel:		x0
"biasadd_readvariableop_conv2d_bias:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�/conv2d/kernel/Regularizer/Square/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOp#conv2d_readvariableop_conv2d_kernel*&
_output_shapes
:		x*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
data_formatNCHW*
paddingVALID*
strides
u
BiasAdd/ReadVariableOpReadVariableOp"biasadd_readvariableop_conv2d_bias*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
data_formatNCHW�
/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp#conv2d_readvariableop_conv2d_kernel*&
_output_shapes
:		x*
dtype0�
 conv2d/kernel/Regularizer/SquareSquare7conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		xx
conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d/kernel/Regularizer/SumSum$conv2d/kernel/Regularizer/Square:y:0(conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
conv2d/kernel/Regularizer/mulMul(conv2d/kernel/Regularizer/mul/x:output:0&conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp0^conv2d/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*2
_input_shapes!
:���������x': : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2b
/conv2d/kernel/Regularizer/Square/ReadVariableOp/conv2d/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������x'
 
_user_specified_nameinputs
�
G
+__inference_activation_1_layer_call_fn_3337

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_activation_1_layer_call_and_return_conditional_losses_1528h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������	"
identityIdentity:output:0*.
_input_shapes
:���������	:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
D
(__inference_reshape_1_layer_call_fn_3459

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_reshape_1_layer_call_and_return_conditional_losses_1577h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*&
_input_shapes
:���������Z:O K
'
_output_shapes
:���������Z
 
_user_specified_nameinputs
�&
�
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1190

inputsN
@assignmovingavg_readvariableop_batch_normalization_1_moving_mean:ZT
Fassignmovingavg_1_readvariableop_batch_normalization_1_moving_variance:ZF
8batchnorm_mul_readvariableop_batch_normalization_1_gamma:ZA
3batchnorm_readvariableop_batch_normalization_1_beta:Z
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
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

:Z�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������Zl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
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
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp@assignmovingavg_readvariableop_batch_normalization_1_moving_mean*
_output_shapes
:Z*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:Zx
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:Z�
AssignMovingAvgAssignSubVariableOp@assignmovingavg_readvariableop_batch_normalization_1_moving_meanAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpFassignmovingavg_1_readvariableop_batch_normalization_1_moving_variance*
_output_shapes
:Z*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:Z~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:Z�
AssignMovingAvg_1AssignSubVariableOpFassignmovingavg_1_readvariableop_batch_normalization_1_moving_varianceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:ZP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:Z�
batchnorm/mul/ReadVariableOpReadVariableOp8batchnorm_mul_readvariableop_batch_normalization_1_gamma*
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
:���������Zh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:Z�
batchnorm/ReadVariableOpReadVariableOp3batchnorm_readvariableop_batch_normalization_1_beta*
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
:���������Zb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������Z�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������Z: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������Z
 
_user_specified_nameinputs
�
B
&__inference_flatten_layer_call_fn_3210

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_1494a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*.
_input_shapes
:���������	:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
�
M__inference_batch_normalization_layer_call_and_return_conditional_losses_3254

inputsK
<batchnorm_readvariableop_batch_normalization_moving_variance:	�E
6batchnorm_mul_readvariableop_batch_normalization_gamma:	�I
:batchnorm_readvariableop_1_batch_normalization_moving_mean:	�B
3batchnorm_readvariableop_2_batch_normalization_beta:	�
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp�
batchnorm/ReadVariableOpReadVariableOp<batchnorm_readvariableop_batch_normalization_moving_variance*
_output_shapes	
:�*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:��
batchnorm/mul/ReadVariableOpReadVariableOp6batchnorm_mul_readvariableop_batch_normalization_gamma*
_output_shapes	
:�*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
batchnorm/ReadVariableOp_1ReadVariableOp:batchnorm_readvariableop_1_batch_normalization_moving_mean*
_output_shapes	
:�*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:��
batchnorm/ReadVariableOp_2ReadVariableOp3batchnorm_readvariableop_2_batch_normalization_beta*
_output_shapes	
:�*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*/
_input_shapes
:����������: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1147

inputsL
>batchnorm_readvariableop_batch_normalization_1_moving_variance:ZF
8batchnorm_mul_readvariableop_batch_normalization_1_gamma:ZJ
<batchnorm_readvariableop_1_batch_normalization_1_moving_mean:ZC
5batchnorm_readvariableop_2_batch_normalization_1_beta:Z
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp�
batchnorm/ReadVariableOpReadVariableOp>batchnorm_readvariableop_batch_normalization_1_moving_variance*
_output_shapes
:Z*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:ZP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:Z�
batchnorm/mul/ReadVariableOpReadVariableOp8batchnorm_mul_readvariableop_batch_normalization_1_gamma*
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
:���������Z�
batchnorm/ReadVariableOp_1ReadVariableOp<batchnorm_readvariableop_1_batch_normalization_1_moving_mean*
_output_shapes
:Z*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:Z�
batchnorm/ReadVariableOp_2ReadVariableOp5batchnorm_readvariableop_2_batch_normalization_1_beta*
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
:���������Zb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������Z�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������Z: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������Z
 
_user_specified_nameinputs
�'
�
M__inference_batch_normalization_layer_call_and_return_conditional_losses_1062

inputsM
>assignmovingavg_readvariableop_batch_normalization_moving_mean:	�S
Dassignmovingavg_1_readvariableop_batch_normalization_moving_variance:	�E
6batchnorm_mul_readvariableop_batch_normalization_gamma:	�@
1batchnorm_readvariableop_batch_normalization_beta:	�
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	��
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp>assignmovingavg_readvariableop_batch_normalization_moving_mean*
_output_shapes	
:�*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:�y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvgAssignSubVariableOp>assignmovingavg_readvariableop_batch_normalization_moving_meanAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpDassignmovingavg_1_readvariableop_batch_normalization_moving_variance*
_output_shapes	
:�*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvg_1AssignSubVariableOpDassignmovingavg_1_readvariableop_batch_normalization_moving_varianceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:��
batchnorm/mul/ReadVariableOpReadVariableOp6batchnorm_mul_readvariableop_batch_normalization_gamma*
_output_shapes	
:�*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:��
batchnorm/ReadVariableOpReadVariableOp1batchnorm_readvariableop_batch_normalization_beta*
_output_shapes	
:�*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
_
C__inference_reshape_1_layer_call_and_return_conditional_losses_3473

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
valueB:�
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
value	B :�
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:���������`
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*&
_input_shapes
:���������Z:O K
'
_output_shapes
:���������Z
 
_user_specified_nameinputs
�
�
B__inference_conv2d_1_layer_call_and_return_conditional_losses_3205

inputs?
%conv2d_readvariableop_conv2d_1_kernel:2
$biasadd_readvariableop_conv2d_1_bias:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�1conv2d_1/kernel/Regularizer/Square/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOp%conv2d_readvariableop_conv2d_1_kernel*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������	*
data_formatNCHW*
paddingVALID*
strides
w
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_conv2d_1_bias*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������	*
data_formatNCHW�
1conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%conv2d_readvariableop_conv2d_1_kernel*&
_output_shapes
:*
dtype0�
"conv2d_1/kernel/Regularizer/SquareSquare9conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
!conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_1/kernel/Regularizer/SumSum&conv2d_1/kernel/Regularizer/Square:y:0*conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
conv2d_1/kernel/Regularizer/mulMul*conv2d_1/kernel/Regularizer/mul/x:output:0(conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:���������	�
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp2^conv2d_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*2
_input_shapes!
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2f
1conv2d_1/kernel/Regularizer/Square/ReadVariableOp1conv2d_1/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
`
D__inference_activation_layer_call_and_return_conditional_losses_1466

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:���������b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
_
C__inference_flatten_2_layer_call_and_return_conditional_losses_3519

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����Z   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:���������ZX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:���������Z"
identityIdentity:output:0*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
$__inference_dense_layer_call_fn_3604

inputs
dense_kernel:Z%

dense_bias:%
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsdense_kernel
dense_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������%*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_1621o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������%`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0**
_input_shapes
:���������Z: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������Z
 
_user_specified_nameinputs
�
G
+__inference_activation_3_layer_call_fn_3618

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������%* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_activation_3_layer_call_and_return_conditional_losses_1650`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������%"
identityIdentity:output:0*&
_input_shapes
:���������%:O K
'
_output_shapes
:���������%
 
_user_specified_nameinputs
��
�
__inference__wrapped_model_978
input_1f
Pcnn_2d_norm_layer_normalization_reshape_readvariableop_layer_normalization_gamma:x'g
Qcnn_2d_norm_layer_normalization_reshape_1_readvariableop_layer_normalization_beta:x'P
6cnn_2d_norm_conv2d_conv2d_readvariableop_conv2d_kernel:		xC
5cnn_2d_norm_conv2d_biasadd_readvariableop_conv2d_bias:T
:cnn_2d_norm_conv2d_1_conv2d_readvariableop_conv2d_1_kernel:G
9cnn_2d_norm_conv2d_1_biasadd_readvariableop_conv2d_1_bias:k
\cnn_2d_norm_batch_normalization_batchnorm_readvariableop_batch_normalization_moving_variance:	�e
Vcnn_2d_norm_batch_normalization_batchnorm_mul_readvariableop_batch_normalization_gamma:	�i
Zcnn_2d_norm_batch_normalization_batchnorm_readvariableop_1_batch_normalization_moving_mean:	�b
Scnn_2d_norm_batch_normalization_batchnorm_readvariableop_2_batch_normalization_beta:	�T
:cnn_2d_norm_conv2d_2_conv2d_readvariableop_conv2d_2_kernel:G
9cnn_2d_norm_conv2d_2_biasadd_readvariableop_conv2d_2_bias:n
`cnn_2d_norm_batch_normalization_1_batchnorm_readvariableop_batch_normalization_1_moving_variance:Zh
Zcnn_2d_norm_batch_normalization_1_batchnorm_mul_readvariableop_batch_normalization_1_gamma:Zl
^cnn_2d_norm_batch_normalization_1_batchnorm_readvariableop_1_batch_normalization_1_moving_mean:Ze
Wcnn_2d_norm_batch_normalization_1_batchnorm_readvariableop_2_batch_normalization_1_beta:Zn
`cnn_2d_norm_batch_normalization_2_batchnorm_readvariableop_batch_normalization_2_moving_variance:Zh
Zcnn_2d_norm_batch_normalization_2_batchnorm_mul_readvariableop_batch_normalization_2_gamma:Zl
^cnn_2d_norm_batch_normalization_2_batchnorm_readvariableop_1_batch_normalization_2_moving_mean:Ze
Wcnn_2d_norm_batch_normalization_2_batchnorm_readvariableop_2_batch_normalization_2_beta:ZF
4cnn_2d_norm_dense_matmul_readvariableop_dense_kernel:Z%A
3cnn_2d_norm_dense_biasadd_readvariableop_dense_bias:%
identity��8CNN_2D_NORM/batch_normalization/batchnorm/ReadVariableOp�:CNN_2D_NORM/batch_normalization/batchnorm/ReadVariableOp_1�:CNN_2D_NORM/batch_normalization/batchnorm/ReadVariableOp_2�<CNN_2D_NORM/batch_normalization/batchnorm/mul/ReadVariableOp�:CNN_2D_NORM/batch_normalization_1/batchnorm/ReadVariableOp�<CNN_2D_NORM/batch_normalization_1/batchnorm/ReadVariableOp_1�<CNN_2D_NORM/batch_normalization_1/batchnorm/ReadVariableOp_2�>CNN_2D_NORM/batch_normalization_1/batchnorm/mul/ReadVariableOp�:CNN_2D_NORM/batch_normalization_2/batchnorm/ReadVariableOp�<CNN_2D_NORM/batch_normalization_2/batchnorm/ReadVariableOp_1�<CNN_2D_NORM/batch_normalization_2/batchnorm/ReadVariableOp_2�>CNN_2D_NORM/batch_normalization_2/batchnorm/mul/ReadVariableOp�)CNN_2D_NORM/conv2d/BiasAdd/ReadVariableOp�(CNN_2D_NORM/conv2d/Conv2D/ReadVariableOp�+CNN_2D_NORM/conv2d_1/BiasAdd/ReadVariableOp�*CNN_2D_NORM/conv2d_1/Conv2D/ReadVariableOp�+CNN_2D_NORM/conv2d_2/BiasAdd/ReadVariableOp�*CNN_2D_NORM/conv2d_2/Conv2D/ReadVariableOp�(CNN_2D_NORM/dense/BiasAdd/ReadVariableOp�'CNN_2D_NORM/dense/MatMul/ReadVariableOp�6CNN_2D_NORM/layer_normalization/Reshape/ReadVariableOp�8CNN_2D_NORM/layer_normalization/Reshape_1/ReadVariableOp�
>CNN_2D_NORM/layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         �
,CNN_2D_NORM/layer_normalization/moments/meanMeaninput_1GCNN_2D_NORM/layer_normalization/moments/mean/reduction_indices:output:0*
T0*/
_output_shapes
:���������*
	keep_dims(�
4CNN_2D_NORM/layer_normalization/moments/StopGradientStopGradient5CNN_2D_NORM/layer_normalization/moments/mean:output:0*
T0*/
_output_shapes
:����������
9CNN_2D_NORM/layer_normalization/moments/SquaredDifferenceSquaredDifferenceinput_1=CNN_2D_NORM/layer_normalization/moments/StopGradient:output:0*
T0*/
_output_shapes
:���������x'�
BCNN_2D_NORM/layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         �
0CNN_2D_NORM/layer_normalization/moments/varianceMean=CNN_2D_NORM/layer_normalization/moments/SquaredDifference:z:0KCNN_2D_NORM/layer_normalization/moments/variance/reduction_indices:output:0*
T0*/
_output_shapes
:���������*
	keep_dims(�
6CNN_2D_NORM/layer_normalization/Reshape/ReadVariableOpReadVariableOpPcnn_2d_norm_layer_normalization_reshape_readvariableop_layer_normalization_gamma*"
_output_shapes
:x'*
dtype0�
-CNN_2D_NORM/layer_normalization/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   x      '   �
'CNN_2D_NORM/layer_normalization/ReshapeReshape>CNN_2D_NORM/layer_normalization/Reshape/ReadVariableOp:value:06CNN_2D_NORM/layer_normalization/Reshape/shape:output:0*
T0*&
_output_shapes
:x'�
8CNN_2D_NORM/layer_normalization/Reshape_1/ReadVariableOpReadVariableOpQcnn_2d_norm_layer_normalization_reshape_1_readvariableop_layer_normalization_beta*"
_output_shapes
:x'*
dtype0�
/CNN_2D_NORM/layer_normalization/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   x      '   �
)CNN_2D_NORM/layer_normalization/Reshape_1Reshape@CNN_2D_NORM/layer_normalization/Reshape_1/ReadVariableOp:value:08CNN_2D_NORM/layer_normalization/Reshape_1/shape:output:0*
T0*&
_output_shapes
:x't
/CNN_2D_NORM/layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
-CNN_2D_NORM/layer_normalization/batchnorm/addAddV29CNN_2D_NORM/layer_normalization/moments/variance:output:08CNN_2D_NORM/layer_normalization/batchnorm/add/y:output:0*
T0*/
_output_shapes
:����������
/CNN_2D_NORM/layer_normalization/batchnorm/RsqrtRsqrt1CNN_2D_NORM/layer_normalization/batchnorm/add:z:0*
T0*/
_output_shapes
:����������
-CNN_2D_NORM/layer_normalization/batchnorm/mulMul3CNN_2D_NORM/layer_normalization/batchnorm/Rsqrt:y:00CNN_2D_NORM/layer_normalization/Reshape:output:0*
T0*/
_output_shapes
:���������x'�
/CNN_2D_NORM/layer_normalization/batchnorm/mul_1Mulinput_11CNN_2D_NORM/layer_normalization/batchnorm/mul:z:0*
T0*/
_output_shapes
:���������x'�
/CNN_2D_NORM/layer_normalization/batchnorm/mul_2Mul5CNN_2D_NORM/layer_normalization/moments/mean:output:01CNN_2D_NORM/layer_normalization/batchnorm/mul:z:0*
T0*/
_output_shapes
:���������x'�
-CNN_2D_NORM/layer_normalization/batchnorm/subSub2CNN_2D_NORM/layer_normalization/Reshape_1:output:03CNN_2D_NORM/layer_normalization/batchnorm/mul_2:z:0*
T0*/
_output_shapes
:���������x'�
/CNN_2D_NORM/layer_normalization/batchnorm/add_1AddV23CNN_2D_NORM/layer_normalization/batchnorm/mul_1:z:01CNN_2D_NORM/layer_normalization/batchnorm/sub:z:0*
T0*/
_output_shapes
:���������x'�
(CNN_2D_NORM/conv2d/Conv2D/ReadVariableOpReadVariableOp6cnn_2d_norm_conv2d_conv2d_readvariableop_conv2d_kernel*&
_output_shapes
:		x*
dtype0�
CNN_2D_NORM/conv2d/Conv2DConv2D3CNN_2D_NORM/layer_normalization/batchnorm/add_1:z:00CNN_2D_NORM/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
data_formatNCHW*
paddingVALID*
strides
�
)CNN_2D_NORM/conv2d/BiasAdd/ReadVariableOpReadVariableOp5cnn_2d_norm_conv2d_biasadd_readvariableop_conv2d_bias*
_output_shapes
:*
dtype0�
CNN_2D_NORM/conv2d/BiasAddBiasAdd"CNN_2D_NORM/conv2d/Conv2D:output:01CNN_2D_NORM/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
data_formatNCHW�
!CNN_2D_NORM/max_pooling2d/MaxPoolMaxPool#CNN_2D_NORM/conv2d/BiasAdd:output:0*/
_output_shapes
:���������*
data_formatNCHW*
ksize
*
paddingVALID*
strides
�
CNN_2D_NORM/activation/ReluRelu*CNN_2D_NORM/max_pooling2d/MaxPool:output:0*
T0*/
_output_shapes
:����������
*CNN_2D_NORM/conv2d_1/Conv2D/ReadVariableOpReadVariableOp:cnn_2d_norm_conv2d_1_conv2d_readvariableop_conv2d_1_kernel*&
_output_shapes
:*
dtype0�
CNN_2D_NORM/conv2d_1/Conv2DConv2D)CNN_2D_NORM/activation/Relu:activations:02CNN_2D_NORM/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������	*
data_formatNCHW*
paddingVALID*
strides
�
+CNN_2D_NORM/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp9cnn_2d_norm_conv2d_1_biasadd_readvariableop_conv2d_1_bias*
_output_shapes
:*
dtype0�
CNN_2D_NORM/conv2d_1/BiasAddBiasAdd$CNN_2D_NORM/conv2d_1/Conv2D:output:03CNN_2D_NORM/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������	*
data_formatNCHWj
CNN_2D_NORM/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  �
CNN_2D_NORM/flatten/ReshapeReshape%CNN_2D_NORM/conv2d_1/BiasAdd:output:0"CNN_2D_NORM/flatten/Const:output:0*
T0*(
_output_shapes
:�����������
8CNN_2D_NORM/batch_normalization/batchnorm/ReadVariableOpReadVariableOp\cnn_2d_norm_batch_normalization_batchnorm_readvariableop_batch_normalization_moving_variance*
_output_shapes	
:�*
dtype0t
/CNN_2D_NORM/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
-CNN_2D_NORM/batch_normalization/batchnorm/addAddV2@CNN_2D_NORM/batch_normalization/batchnorm/ReadVariableOp:value:08CNN_2D_NORM/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
/CNN_2D_NORM/batch_normalization/batchnorm/RsqrtRsqrt1CNN_2D_NORM/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:��
<CNN_2D_NORM/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOpVcnn_2d_norm_batch_normalization_batchnorm_mul_readvariableop_batch_normalization_gamma*
_output_shapes	
:�*
dtype0�
-CNN_2D_NORM/batch_normalization/batchnorm/mulMul3CNN_2D_NORM/batch_normalization/batchnorm/Rsqrt:y:0DCNN_2D_NORM/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
/CNN_2D_NORM/batch_normalization/batchnorm/mul_1Mul$CNN_2D_NORM/flatten/Reshape:output:01CNN_2D_NORM/batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
:CNN_2D_NORM/batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOpZcnn_2d_norm_batch_normalization_batchnorm_readvariableop_1_batch_normalization_moving_mean*
_output_shapes	
:�*
dtype0�
/CNN_2D_NORM/batch_normalization/batchnorm/mul_2MulBCNN_2D_NORM/batch_normalization/batchnorm/ReadVariableOp_1:value:01CNN_2D_NORM/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
:CNN_2D_NORM/batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOpScnn_2d_norm_batch_normalization_batchnorm_readvariableop_2_batch_normalization_beta*
_output_shapes	
:�*
dtype0�
-CNN_2D_NORM/batch_normalization/batchnorm/subSubBCNN_2D_NORM/batch_normalization/batchnorm/ReadVariableOp_2:value:03CNN_2D_NORM/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
/CNN_2D_NORM/batch_normalization/batchnorm/add_1AddV23CNN_2D_NORM/batch_normalization/batchnorm/mul_1:z:01CNN_2D_NORM/batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������|
CNN_2D_NORM/reshape/ShapeShape3CNN_2D_NORM/batch_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:q
'CNN_2D_NORM/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)CNN_2D_NORM/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)CNN_2D_NORM/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
!CNN_2D_NORM/reshape/strided_sliceStridedSlice"CNN_2D_NORM/reshape/Shape:output:00CNN_2D_NORM/reshape/strided_slice/stack:output:02CNN_2D_NORM/reshape/strided_slice/stack_1:output:02CNN_2D_NORM/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#CNN_2D_NORM/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :e
#CNN_2D_NORM/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :e
#CNN_2D_NORM/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :	�
!CNN_2D_NORM/reshape/Reshape/shapePack*CNN_2D_NORM/reshape/strided_slice:output:0,CNN_2D_NORM/reshape/Reshape/shape/1:output:0,CNN_2D_NORM/reshape/Reshape/shape/2:output:0,CNN_2D_NORM/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
CNN_2D_NORM/reshape/ReshapeReshape3CNN_2D_NORM/batch_normalization/batchnorm/add_1:z:0*CNN_2D_NORM/reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:���������	�
CNN_2D_NORM/activation_1/ReluRelu$CNN_2D_NORM/reshape/Reshape:output:0*
T0*/
_output_shapes
:���������	�
*CNN_2D_NORM/conv2d_2/Conv2D/ReadVariableOpReadVariableOp:cnn_2d_norm_conv2d_2_conv2d_readvariableop_conv2d_2_kernel*&
_output_shapes
:*
dtype0�
CNN_2D_NORM/conv2d_2/Conv2DConv2D+CNN_2D_NORM/activation_1/Relu:activations:02CNN_2D_NORM/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
data_formatNCHW*
paddingVALID*
strides
�
+CNN_2D_NORM/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp9cnn_2d_norm_conv2d_2_biasadd_readvariableop_conv2d_2_bias*
_output_shapes
:*
dtype0�
CNN_2D_NORM/conv2d_2/BiasAddBiasAdd$CNN_2D_NORM/conv2d_2/Conv2D:output:03CNN_2D_NORM/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
data_formatNCHWl
CNN_2D_NORM/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
CNN_2D_NORM/flatten_1/ReshapeReshape%CNN_2D_NORM/conv2d_2/BiasAdd:output:0$CNN_2D_NORM/flatten_1/Const:output:0*
T0*'
_output_shapes
:���������Z�
:CNN_2D_NORM/batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp`cnn_2d_norm_batch_normalization_1_batchnorm_readvariableop_batch_normalization_1_moving_variance*
_output_shapes
:Z*
dtype0v
1CNN_2D_NORM/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
/CNN_2D_NORM/batch_normalization_1/batchnorm/addAddV2BCNN_2D_NORM/batch_normalization_1/batchnorm/ReadVariableOp:value:0:CNN_2D_NORM/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes
:Z�
1CNN_2D_NORM/batch_normalization_1/batchnorm/RsqrtRsqrt3CNN_2D_NORM/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
:Z�
>CNN_2D_NORM/batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpZcnn_2d_norm_batch_normalization_1_batchnorm_mul_readvariableop_batch_normalization_1_gamma*
_output_shapes
:Z*
dtype0�
/CNN_2D_NORM/batch_normalization_1/batchnorm/mulMul5CNN_2D_NORM/batch_normalization_1/batchnorm/Rsqrt:y:0FCNN_2D_NORM/batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Z�
1CNN_2D_NORM/batch_normalization_1/batchnorm/mul_1Mul&CNN_2D_NORM/flatten_1/Reshape:output:03CNN_2D_NORM/batch_normalization_1/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������Z�
<CNN_2D_NORM/batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOp^cnn_2d_norm_batch_normalization_1_batchnorm_readvariableop_1_batch_normalization_1_moving_mean*
_output_shapes
:Z*
dtype0�
1CNN_2D_NORM/batch_normalization_1/batchnorm/mul_2MulDCNN_2D_NORM/batch_normalization_1/batchnorm/ReadVariableOp_1:value:03CNN_2D_NORM/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
:Z�
<CNN_2D_NORM/batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOpWcnn_2d_norm_batch_normalization_1_batchnorm_readvariableop_2_batch_normalization_1_beta*
_output_shapes
:Z*
dtype0�
/CNN_2D_NORM/batch_normalization_1/batchnorm/subSubDCNN_2D_NORM/batch_normalization_1/batchnorm/ReadVariableOp_2:value:05CNN_2D_NORM/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Z�
1CNN_2D_NORM/batch_normalization_1/batchnorm/add_1AddV25CNN_2D_NORM/batch_normalization_1/batchnorm/mul_1:z:03CNN_2D_NORM/batch_normalization_1/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������Z�
CNN_2D_NORM/reshape_1/ShapeShape5CNN_2D_NORM/batch_normalization_1/batchnorm/add_1:z:0*
T0*
_output_shapes
:s
)CNN_2D_NORM/reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+CNN_2D_NORM/reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+CNN_2D_NORM/reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
#CNN_2D_NORM/reshape_1/strided_sliceStridedSlice$CNN_2D_NORM/reshape_1/Shape:output:02CNN_2D_NORM/reshape_1/strided_slice/stack:output:04CNN_2D_NORM/reshape_1/strided_slice/stack_1:output:04CNN_2D_NORM/reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
%CNN_2D_NORM/reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :g
%CNN_2D_NORM/reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :g
%CNN_2D_NORM/reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
#CNN_2D_NORM/reshape_1/Reshape/shapePack,CNN_2D_NORM/reshape_1/strided_slice:output:0.CNN_2D_NORM/reshape_1/Reshape/shape/1:output:0.CNN_2D_NORM/reshape_1/Reshape/shape/2:output:0.CNN_2D_NORM/reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
CNN_2D_NORM/reshape_1/ReshapeReshape5CNN_2D_NORM/batch_normalization_1/batchnorm/add_1:z:0,CNN_2D_NORM/reshape_1/Reshape/shape:output:0*
T0*/
_output_shapes
:����������
CNN_2D_NORM/activation_2/ReluRelu&CNN_2D_NORM/reshape_1/Reshape:output:0*
T0*/
_output_shapes
:���������l
CNN_2D_NORM/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"����Z   �
CNN_2D_NORM/flatten_2/ReshapeReshape+CNN_2D_NORM/activation_2/Relu:activations:0$CNN_2D_NORM/flatten_2/Const:output:0*
T0*'
_output_shapes
:���������Z�
:CNN_2D_NORM/batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp`cnn_2d_norm_batch_normalization_2_batchnorm_readvariableop_batch_normalization_2_moving_variance*
_output_shapes
:Z*
dtype0v
1CNN_2D_NORM/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
/CNN_2D_NORM/batch_normalization_2/batchnorm/addAddV2BCNN_2D_NORM/batch_normalization_2/batchnorm/ReadVariableOp:value:0:CNN_2D_NORM/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:Z�
1CNN_2D_NORM/batch_normalization_2/batchnorm/RsqrtRsqrt3CNN_2D_NORM/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:Z�
>CNN_2D_NORM/batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpZcnn_2d_norm_batch_normalization_2_batchnorm_mul_readvariableop_batch_normalization_2_gamma*
_output_shapes
:Z*
dtype0�
/CNN_2D_NORM/batch_normalization_2/batchnorm/mulMul5CNN_2D_NORM/batch_normalization_2/batchnorm/Rsqrt:y:0FCNN_2D_NORM/batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Z�
1CNN_2D_NORM/batch_normalization_2/batchnorm/mul_1Mul&CNN_2D_NORM/flatten_2/Reshape:output:03CNN_2D_NORM/batch_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������Z�
<CNN_2D_NORM/batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOp^cnn_2d_norm_batch_normalization_2_batchnorm_readvariableop_1_batch_normalization_2_moving_mean*
_output_shapes
:Z*
dtype0�
1CNN_2D_NORM/batch_normalization_2/batchnorm/mul_2MulDCNN_2D_NORM/batch_normalization_2/batchnorm/ReadVariableOp_1:value:03CNN_2D_NORM/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:Z�
<CNN_2D_NORM/batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOpWcnn_2d_norm_batch_normalization_2_batchnorm_readvariableop_2_batch_normalization_2_beta*
_output_shapes
:Z*
dtype0�
/CNN_2D_NORM/batch_normalization_2/batchnorm/subSubDCNN_2D_NORM/batch_normalization_2/batchnorm/ReadVariableOp_2:value:05CNN_2D_NORM/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Z�
1CNN_2D_NORM/batch_normalization_2/batchnorm/add_1AddV25CNN_2D_NORM/batch_normalization_2/batchnorm/mul_1:z:03CNN_2D_NORM/batch_normalization_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������Z�
'CNN_2D_NORM/dense/MatMul/ReadVariableOpReadVariableOp4cnn_2d_norm_dense_matmul_readvariableop_dense_kernel*
_output_shapes

:Z%*
dtype0�
CNN_2D_NORM/dense/MatMulMatMul5CNN_2D_NORM/batch_normalization_2/batchnorm/add_1:z:0/CNN_2D_NORM/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������%�
(CNN_2D_NORM/dense/BiasAdd/ReadVariableOpReadVariableOp3cnn_2d_norm_dense_biasadd_readvariableop_dense_bias*
_output_shapes
:%*
dtype0�
CNN_2D_NORM/dense/BiasAddBiasAdd"CNN_2D_NORM/dense/MatMul:product:00CNN_2D_NORM/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������%�
)CNN_2D_NORM/dense/ActivityRegularizer/AbsAbs"CNN_2D_NORM/dense/BiasAdd:output:0*
T0*'
_output_shapes
:���������%|
+CNN_2D_NORM/dense/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
)CNN_2D_NORM/dense/ActivityRegularizer/SumSum-CNN_2D_NORM/dense/ActivityRegularizer/Abs:y:04CNN_2D_NORM/dense/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: p
+CNN_2D_NORM/dense/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
)CNN_2D_NORM/dense/ActivityRegularizer/mulMul4CNN_2D_NORM/dense/ActivityRegularizer/mul/x:output:02CNN_2D_NORM/dense/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: }
+CNN_2D_NORM/dense/ActivityRegularizer/ShapeShape"CNN_2D_NORM/dense/BiasAdd:output:0*
T0*
_output_shapes
:�
9CNN_2D_NORM/dense/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
;CNN_2D_NORM/dense/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
;CNN_2D_NORM/dense/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
3CNN_2D_NORM/dense/ActivityRegularizer/strided_sliceStridedSlice4CNN_2D_NORM/dense/ActivityRegularizer/Shape:output:0BCNN_2D_NORM/dense/ActivityRegularizer/strided_slice/stack:output:0DCNN_2D_NORM/dense/ActivityRegularizer/strided_slice/stack_1:output:0DCNN_2D_NORM/dense/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
*CNN_2D_NORM/dense/ActivityRegularizer/CastCast<CNN_2D_NORM/dense/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: �
-CNN_2D_NORM/dense/ActivityRegularizer/truedivRealDiv-CNN_2D_NORM/dense/ActivityRegularizer/mul:z:0.CNN_2D_NORM/dense/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: �
!CNN_2D_NORM/activation_3/SoftplusSoftplus"CNN_2D_NORM/dense/BiasAdd:output:0*
T0*'
_output_shapes
:���������%~
IdentityIdentity/CNN_2D_NORM/activation_3/Softplus:activations:0^NoOp*
T0*'
_output_shapes
:���������%�

NoOpNoOp9^CNN_2D_NORM/batch_normalization/batchnorm/ReadVariableOp;^CNN_2D_NORM/batch_normalization/batchnorm/ReadVariableOp_1;^CNN_2D_NORM/batch_normalization/batchnorm/ReadVariableOp_2=^CNN_2D_NORM/batch_normalization/batchnorm/mul/ReadVariableOp;^CNN_2D_NORM/batch_normalization_1/batchnorm/ReadVariableOp=^CNN_2D_NORM/batch_normalization_1/batchnorm/ReadVariableOp_1=^CNN_2D_NORM/batch_normalization_1/batchnorm/ReadVariableOp_2?^CNN_2D_NORM/batch_normalization_1/batchnorm/mul/ReadVariableOp;^CNN_2D_NORM/batch_normalization_2/batchnorm/ReadVariableOp=^CNN_2D_NORM/batch_normalization_2/batchnorm/ReadVariableOp_1=^CNN_2D_NORM/batch_normalization_2/batchnorm/ReadVariableOp_2?^CNN_2D_NORM/batch_normalization_2/batchnorm/mul/ReadVariableOp*^CNN_2D_NORM/conv2d/BiasAdd/ReadVariableOp)^CNN_2D_NORM/conv2d/Conv2D/ReadVariableOp,^CNN_2D_NORM/conv2d_1/BiasAdd/ReadVariableOp+^CNN_2D_NORM/conv2d_1/Conv2D/ReadVariableOp,^CNN_2D_NORM/conv2d_2/BiasAdd/ReadVariableOp+^CNN_2D_NORM/conv2d_2/Conv2D/ReadVariableOp)^CNN_2D_NORM/dense/BiasAdd/ReadVariableOp(^CNN_2D_NORM/dense/MatMul/ReadVariableOp7^CNN_2D_NORM/layer_normalization/Reshape/ReadVariableOp9^CNN_2D_NORM/layer_normalization/Reshape_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*Z
_input_shapesI
G:���������x': : : : : : : : : : : : : : : : : : : : : : 2t
8CNN_2D_NORM/batch_normalization/batchnorm/ReadVariableOp8CNN_2D_NORM/batch_normalization/batchnorm/ReadVariableOp2x
:CNN_2D_NORM/batch_normalization/batchnorm/ReadVariableOp_1:CNN_2D_NORM/batch_normalization/batchnorm/ReadVariableOp_12x
:CNN_2D_NORM/batch_normalization/batchnorm/ReadVariableOp_2:CNN_2D_NORM/batch_normalization/batchnorm/ReadVariableOp_22|
<CNN_2D_NORM/batch_normalization/batchnorm/mul/ReadVariableOp<CNN_2D_NORM/batch_normalization/batchnorm/mul/ReadVariableOp2x
:CNN_2D_NORM/batch_normalization_1/batchnorm/ReadVariableOp:CNN_2D_NORM/batch_normalization_1/batchnorm/ReadVariableOp2|
<CNN_2D_NORM/batch_normalization_1/batchnorm/ReadVariableOp_1<CNN_2D_NORM/batch_normalization_1/batchnorm/ReadVariableOp_12|
<CNN_2D_NORM/batch_normalization_1/batchnorm/ReadVariableOp_2<CNN_2D_NORM/batch_normalization_1/batchnorm/ReadVariableOp_22�
>CNN_2D_NORM/batch_normalization_1/batchnorm/mul/ReadVariableOp>CNN_2D_NORM/batch_normalization_1/batchnorm/mul/ReadVariableOp2x
:CNN_2D_NORM/batch_normalization_2/batchnorm/ReadVariableOp:CNN_2D_NORM/batch_normalization_2/batchnorm/ReadVariableOp2|
<CNN_2D_NORM/batch_normalization_2/batchnorm/ReadVariableOp_1<CNN_2D_NORM/batch_normalization_2/batchnorm/ReadVariableOp_12|
<CNN_2D_NORM/batch_normalization_2/batchnorm/ReadVariableOp_2<CNN_2D_NORM/batch_normalization_2/batchnorm/ReadVariableOp_22�
>CNN_2D_NORM/batch_normalization_2/batchnorm/mul/ReadVariableOp>CNN_2D_NORM/batch_normalization_2/batchnorm/mul/ReadVariableOp2V
)CNN_2D_NORM/conv2d/BiasAdd/ReadVariableOp)CNN_2D_NORM/conv2d/BiasAdd/ReadVariableOp2T
(CNN_2D_NORM/conv2d/Conv2D/ReadVariableOp(CNN_2D_NORM/conv2d/Conv2D/ReadVariableOp2Z
+CNN_2D_NORM/conv2d_1/BiasAdd/ReadVariableOp+CNN_2D_NORM/conv2d_1/BiasAdd/ReadVariableOp2X
*CNN_2D_NORM/conv2d_1/Conv2D/ReadVariableOp*CNN_2D_NORM/conv2d_1/Conv2D/ReadVariableOp2Z
+CNN_2D_NORM/conv2d_2/BiasAdd/ReadVariableOp+CNN_2D_NORM/conv2d_2/BiasAdd/ReadVariableOp2X
*CNN_2D_NORM/conv2d_2/Conv2D/ReadVariableOp*CNN_2D_NORM/conv2d_2/Conv2D/ReadVariableOp2T
(CNN_2D_NORM/dense/BiasAdd/ReadVariableOp(CNN_2D_NORM/dense/BiasAdd/ReadVariableOp2R
'CNN_2D_NORM/dense/MatMul/ReadVariableOp'CNN_2D_NORM/dense/MatMul/ReadVariableOp2p
6CNN_2D_NORM/layer_normalization/Reshape/ReadVariableOp6CNN_2D_NORM/layer_normalization/Reshape/ReadVariableOp2t
8CNN_2D_NORM/layer_normalization/Reshape_1/ReadVariableOp8CNN_2D_NORM/layer_normalization/Reshape_1/ReadVariableOp:X T
/
_output_shapes
:���������x'
!
_user_specified_name	input_1
�
i
J__inference_gaussian_noise_1_layer_call_and_return_conditional_losses_3332

inputs
identity�;
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
 *���=�
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*/
_output_shapes
:���������	*
dtype0�
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*/
_output_shapes
:���������	�
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*/
_output_shapes
:���������	a
addAddV2inputsrandom_normal:z:0*
T0*/
_output_shapes
:���������	W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:���������	"
identityIdentity:output:0*.
_input_shapes
:���������	:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
K
/__inference_gaussian_noise_1_layer_call_fn_3312

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *S
fNRL
J__inference_gaussian_noise_1_layer_call_and_return_conditional_losses_1521h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������	"
identityIdentity:output:0*.
_input_shapes
:���������	:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
�
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3557

inputsL
>batchnorm_readvariableop_batch_normalization_2_moving_variance:ZF
8batchnorm_mul_readvariableop_batch_normalization_2_gamma:ZJ
<batchnorm_readvariableop_1_batch_normalization_2_moving_mean:ZC
5batchnorm_readvariableop_2_batch_normalization_2_beta:Z
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp�
batchnorm/ReadVariableOpReadVariableOp>batchnorm_readvariableop_batch_normalization_2_moving_variance*
_output_shapes
:Z*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:ZP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:Z�
batchnorm/mul/ReadVariableOpReadVariableOp8batchnorm_mul_readvariableop_batch_normalization_2_gamma*
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
:���������Z�
batchnorm/ReadVariableOp_1ReadVariableOp<batchnorm_readvariableop_1_batch_normalization_2_moving_mean*
_output_shapes
:Z*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:Z�
batchnorm/ReadVariableOp_2ReadVariableOp5batchnorm_readvariableop_2_batch_normalization_2_beta*
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
:���������Zb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������Z�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*.
_input_shapes
:���������Z: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������Z
 
_user_specified_nameinputs
�
�
M__inference_layer_normalization_layer_call_and_return_conditional_losses_1425

inputsF
0reshape_readvariableop_layer_normalization_gamma:x'G
1reshape_1_readvariableop_layer_normalization_beta:x'
identity��Reshape/ReadVariableOp�Reshape_1/ReadVariableOps
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*/
_output_shapes
:���������*
	keep_dims(u
moments/StopGradientStopGradientmoments/mean:output:0*
T0*/
_output_shapes
:����������
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*/
_output_shapes
:���������x'w
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*/
_output_shapes
:���������*
	keep_dims(�
Reshape/ReadVariableOpReadVariableOp0reshape_readvariableop_layer_normalization_gamma*"
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
:x'�
Reshape_1/ReadVariableOpReadVariableOp1reshape_1_readvariableop_layer_normalization_beta*"
_output_shapes
:x'*
dtype0h
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   x      '   �
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:x'T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*/
_output_shapes
:���������e
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*/
_output_shapes
:���������u
batchnorm/mulMulbatchnorm/Rsqrt:y:0Reshape:output:0*
T0*/
_output_shapes
:���������x'k
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*/
_output_shapes
:���������x'z
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*/
_output_shapes
:���������x'w
batchnorm/subSubReshape_1:output:0batchnorm/mul_2:z:0*
T0*/
_output_shapes
:���������x'z
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*/
_output_shapes
:���������x'j
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*/
_output_shapes
:���������x'z
NoOpNoOp^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������x': : 20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:W S
/
_output_shapes
:���������x'
 
_user_specified_nameinputs
�
_
C__inference_flatten_1_layer_call_and_return_conditional_losses_3382

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����Z   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:���������ZX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:���������Z"
identityIdentity:output:0*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
b
F__inference_activation_3_layer_call_and_return_conditional_losses_1650

inputs
identityN
SoftplusSoftplusinputs*
T0*'
_output_shapes
:���������%^
IdentityIdentitySoftplus:activations:0*
T0*'
_output_shapes
:���������%"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������%:O K
'
_output_shapes
:���������%
 
_user_specified_nameinputs
�
g
H__inference_gaussian_noise_layer_call_and_return_conditional_losses_3166

inputs
identity�;
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
 *���=�
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*/
_output_shapes
:���������*
dtype0�
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*/
_output_shapes
:����������
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*/
_output_shapes
:���������a
addAddV2inputsrandom_normal:z:0*
T0*/
_output_shapes
:���������W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
h
/__inference_gaussian_noise_1_layer_call_fn_3317

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *S
fNRL
J__inference_gaussian_noise_1_layer_call_and_return_conditional_losses_1893w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*.
_input_shapes
:���������	22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
��
�
E__inference_CNN_2D_NORM_layer_call_and_return_conditional_losses_1677

inputsC
-layer_normalization_layer_normalization_gamma:x'B
,layer_normalization_layer_normalization_beta:x'.
conv2d_conv2d_kernel:		x 
conv2d_conv2d_bias:2
conv2d_1_conv2d_1_kernel:$
conv2d_1_conv2d_1_bias:F
7batch_normalization_batch_normalization_moving_variance:	�<
-batch_normalization_batch_normalization_gamma:	�B
3batch_normalization_batch_normalization_moving_mean:	�;
,batch_normalization_batch_normalization_beta:	�2
conv2d_2_conv2d_2_kernel:$
conv2d_2_conv2d_2_bias:I
;batch_normalization_1_batch_normalization_1_moving_variance:Z?
1batch_normalization_1_batch_normalization_1_gamma:ZE
7batch_normalization_1_batch_normalization_1_moving_mean:Z>
0batch_normalization_1_batch_normalization_1_beta:ZI
;batch_normalization_2_batch_normalization_2_moving_variance:Z?
1batch_normalization_2_batch_normalization_2_gamma:ZE
7batch_normalization_2_batch_normalization_2_moving_mean:Z>
0batch_normalization_2_batch_normalization_2_beta:Z$
dense_dense_kernel:Z%
dense_dense_bias:%
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�-batch_normalization_2/StatefulPartitionedCall�conv2d/StatefulPartitionedCall�/conv2d/kernel/Regularizer/Square/ReadVariableOp� conv2d_1/StatefulPartitionedCall�1conv2d_1/kernel/Regularizer/Square/ReadVariableOp� conv2d_2/StatefulPartitionedCall�1conv2d_2/kernel/Regularizer/Square/ReadVariableOp�dense/StatefulPartitionedCall�.dense/kernel/Regularizer/Square/ReadVariableOp�+layer_normalization/StatefulPartitionedCall�
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCallinputs-layer_normalization_layer_normalization_gamma,layer_normalization_layer_normalization_beta*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������x'*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *V
fQRO
M__inference_layer_normalization_layer_call_and_return_conditional_losses_1425�
conv2d/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0conv2d_conv2d_kernelconv2d_conv2d_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_1445�
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_1453�
gaussian_noise/PartitionedCallPartitionedCall&max_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_gaussian_noise_layer_call_and_return_conditional_losses_1459�
activation/PartitionedCallPartitionedCall'gaussian_noise/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_activation_layer_call_and_return_conditional_losses_1466�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0conv2d_1_conv2d_1_kernelconv2d_1_conv2d_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_1484�
flatten/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_1494�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:07batch_normalization_batch_normalization_moving_variance-batch_normalization_batch_normalization_gamma3batch_normalization_batch_normalization_moving_mean,batch_normalization_batch_normalization_beta*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_1019�
reshape/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_reshape_layer_call_and_return_conditional_losses_1515�
 gaussian_noise_1/PartitionedCallPartitionedCall reshape/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *S
fNRL
J__inference_gaussian_noise_1_layer_call_and_return_conditional_losses_1521�
activation_1/PartitionedCallPartitionedCall)gaussian_noise_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_activation_1_layer_call_and_return_conditional_losses_1528�
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0conv2d_2_conv2d_2_kernelconv2d_2_conv2d_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_1546�
flatten_1/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������Z* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_flatten_1_layer_call_and_return_conditional_losses_1556�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0;batch_normalization_1_batch_normalization_1_moving_variance1batch_normalization_1_batch_normalization_1_gamma7batch_normalization_1_batch_normalization_1_moving_mean0batch_normalization_1_batch_normalization_1_beta*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������Z*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1147�
reshape_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_reshape_1_layer_call_and_return_conditional_losses_1577�
 gaussian_noise_2/PartitionedCallPartitionedCall"reshape_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *S
fNRL
J__inference_gaussian_noise_2_layer_call_and_return_conditional_losses_1583�
activation_2/PartitionedCallPartitionedCall)gaussian_noise_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_activation_2_layer_call_and_return_conditional_losses_1590�
flatten_2/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������Z* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_flatten_2_layer_call_and_return_conditional_losses_1598�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0;batch_normalization_2_batch_normalization_2_moving_variance1batch_normalization_2_batch_normalization_2_gamma7batch_normalization_2_batch_normalization_2_moving_mean0batch_normalization_2_batch_normalization_2_beta*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������Z*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1275�
dense/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0dense_dense_kerneldense_dense_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������%*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_1621�
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
 *2
config_proto" 

CPU

GPU2 *0J 8� *4
f/R-
+__inference_dense_activity_regularizer_1636u
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
valueB:�
'dense/ActivityRegularizer/strided_sliceStridedSlice(dense/ActivityRegularizer/Shape:output:06dense/ActivityRegularizer/strided_slice/stack:output:08dense/ActivityRegularizer/strided_slice/stack_1:output:08dense/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
dense/ActivityRegularizer/CastCast0dense/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: �
!dense/ActivityRegularizer/truedivRealDiv2dense/ActivityRegularizer/PartitionedCall:output:0"dense/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: �
activation_3/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������%* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_activation_3_layer_call_and_return_conditional_losses_1650�
/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_conv2d_kernel*&
_output_shapes
:		x*
dtype0�
 conv2d/kernel/Regularizer/SquareSquare7conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		xx
conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d/kernel/Regularizer/SumSum$conv2d/kernel/Regularizer/Square:y:0(conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
conv2d/kernel/Regularizer/mulMul(conv2d/kernel/Regularizer/mul/x:output:0&conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_1_conv2d_1_kernel*&
_output_shapes
:*
dtype0�
"conv2d_1/kernel/Regularizer/SquareSquare9conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
!conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_1/kernel/Regularizer/SumSum&conv2d_1/kernel/Regularizer/Square:y:0*conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
conv2d_1/kernel/Regularizer/mulMul*conv2d_1/kernel/Regularizer/mul/x:output:0(conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_2_conv2d_2_kernel*&
_output_shapes
:*
dtype0�
"conv2d_2/kernel/Regularizer/SquareSquare9conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
!conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_2/kernel/Regularizer/SumSum&conv2d_2/kernel/Regularizer/Square:y:0*conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
conv2d_2/kernel/Regularizer/mulMul*conv2d_2/kernel/Regularizer/mul/x:output:0(conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_dense_kernel*
_output_shapes

:Z%*
dtype0�
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:Z%o
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: t
IdentityIdentity%activation_3/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������%�
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall^conv2d/StatefulPartitionedCall0^conv2d/kernel/Regularizer/Square/ReadVariableOp!^conv2d_1/StatefulPartitionedCall2^conv2d_1/kernel/Regularizer/Square/ReadVariableOp!^conv2d_2/StatefulPartitionedCall2^conv2d_2/kernel/Regularizer/Square/ReadVariableOp^dense/StatefulPartitionedCall/^dense/kernel/Regularizer/Square/ReadVariableOp,^layer_normalization/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������x': : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2b
/conv2d/kernel/Regularizer/Square/ReadVariableOp/conv2d/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2f
1conv2d_1/kernel/Regularizer/Square/ReadVariableOp1conv2d_1/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2f
1conv2d_2/kernel/Regularizer/Square/ReadVariableOp1conv2d_2/kernel/Regularizer/Square/ReadVariableOp2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall:W S
/
_output_shapes
:���������x'
 
_user_specified_nameinputs
�
f
J__inference_gaussian_noise_2_layer_call_and_return_conditional_losses_1583

inputs
identityV
IdentityIdentityinputs*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
_
C__inference_reshape_1_layer_call_and_return_conditional_losses_1577

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
valueB:�
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
value	B :�
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:���������`
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������Z:O K
'
_output_shapes
:���������Z
 
_user_specified_nameinputs
�
G
+__inference_activation_2_layer_call_fn_3503

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_activation_2_layer_call_and_return_conditional_losses_1590h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
H
,__inference_max_pooling2d_layer_call_fn_3131

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_1453h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
f
J__inference_gaussian_noise_1_layer_call_and_return_conditional_losses_3321

inputs
identityV
IdentityIdentityinputs*
T0*/
_output_shapes
:���������	"
identityIdentity:output:0*.
_input_shapes
:���������	:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
�
?__inference_dense_layer_call_and_return_conditional_losses_3683

inputs4
"matmul_readvariableop_dense_kernel:Z%/
!biasadd_readvariableop_dense_bias:%
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�.dense/kernel/Regularizer/Square/ReadVariableOpx
MatMul/ReadVariableOpReadVariableOp"matmul_readvariableop_dense_kernel*
_output_shapes

:Z%*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������%t
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_dense_bias*
_output_shapes
:%*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������%�
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp"matmul_readvariableop_dense_kernel*
_output_shapes

:Z%*
dtype0�
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:Z%o
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������%�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0**
_input_shapes
:���������Z: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������Z
 
_user_specified_nameinputs
�
�
M__inference_layer_normalization_layer_call_and_return_conditional_losses_3092

inputsF
0reshape_readvariableop_layer_normalization_gamma:x'G
1reshape_1_readvariableop_layer_normalization_beta:x'
identity��Reshape/ReadVariableOp�Reshape_1/ReadVariableOps
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*/
_output_shapes
:���������*
	keep_dims(u
moments/StopGradientStopGradientmoments/mean:output:0*
T0*/
_output_shapes
:����������
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*/
_output_shapes
:���������x'w
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*/
_output_shapes
:���������*
	keep_dims(�
Reshape/ReadVariableOpReadVariableOp0reshape_readvariableop_layer_normalization_gamma*"
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
:x'�
Reshape_1/ReadVariableOpReadVariableOp1reshape_1_readvariableop_layer_normalization_beta*"
_output_shapes
:x'*
dtype0h
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   x      '   �
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:x'T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*/
_output_shapes
:���������e
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*/
_output_shapes
:���������u
batchnorm/mulMulbatchnorm/Rsqrt:y:0Reshape:output:0*
T0*/
_output_shapes
:���������x'k
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*/
_output_shapes
:���������x'z
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*/
_output_shapes
:���������x'w
batchnorm/subSubReshape_1:output:0batchnorm/mul_2:z:0*
T0*/
_output_shapes
:���������x'z
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*/
_output_shapes
:���������x'j
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*/
_output_shapes
:���������x'z
NoOpNoOp^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*2
_input_shapes!
:���������x': : 20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:W S
/
_output_shapes
:���������x'
 
_user_specified_nameinputs
�
b
F__inference_activation_1_layer_call_and_return_conditional_losses_1528

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:���������	b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������	:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
�
*__inference_CNN_2D_NORM_layer_call_fn_2639

inputs/
layer_normalization_gamma:x'.
layer_normalization_beta:x''
conv2d_kernel:		x
conv2d_bias:)
conv2d_1_kernel:
conv2d_1_bias:2
#batch_normalization_moving_variance:	�(
batch_normalization_gamma:	�.
batch_normalization_moving_mean:	�'
batch_normalization_beta:	�)
conv2d_2_kernel:
conv2d_2_bias:3
%batch_normalization_1_moving_variance:Z)
batch_normalization_1_gamma:Z/
!batch_normalization_1_moving_mean:Z(
batch_normalization_1_beta:Z3
%batch_normalization_2_moving_variance:Z)
batch_normalization_2_gamma:Z/
!batch_normalization_2_moving_mean:Z(
batch_normalization_2_beta:Z
dense_kernel:Z%

dense_bias:%
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputslayer_normalization_gammalayer_normalization_betaconv2d_kernelconv2d_biasconv2d_1_kernelconv2d_1_bias#batch_normalization_moving_variancebatch_normalization_gammabatch_normalization_moving_meanbatch_normalization_betaconv2d_2_kernelconv2d_2_bias%batch_normalization_1_moving_variancebatch_normalization_1_gamma!batch_normalization_1_moving_meanbatch_normalization_1_beta%batch_normalization_2_moving_variancebatch_normalization_2_gamma!batch_normalization_2_moving_meanbatch_normalization_2_betadense_kernel
dense_bias*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������%*8
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_CNN_2D_NORM_layer_call_and_return_conditional_losses_1677o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������%`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*Z
_input_shapesI
G:���������x': : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������x'
 
_user_specified_nameinputs
�	
g
H__inference_gaussian_noise_layer_call_and_return_conditional_losses_1997

inputs
identity�;
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
 *���=�
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*/
_output_shapes
:���������*
dtype0�
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*/
_output_shapes
:����������
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*/
_output_shapes
:���������a
addAddV2inputsrandom_normal:z:0*
T0*/
_output_shapes
:���������W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
__inference_loss_fn_3_3667M
;dense_kernel_regularizer_square_readvariableop_dense_kernel:Z%
identity��.dense/kernel/Regularizer/Square/ReadVariableOp�
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;dense_kernel_regularizer_square_readvariableop_dense_kernel*
_output_shapes

:Z%*
dtype0�
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:Z%o
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
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
identityIdentity:output:0*
_input_shapes
: 2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp
�
�
__inference_loss_fn_0_3634W
=conv2d_kernel_regularizer_square_readvariableop_conv2d_kernel:		x
identity��/conv2d/kernel/Regularizer/Square/ReadVariableOp�
/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp=conv2d_kernel_regularizer_square_readvariableop_conv2d_kernel*&
_output_shapes
:		x*
dtype0�
 conv2d/kernel/Regularizer/SquareSquare7conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		xx
conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d/kernel/Regularizer/SumSum$conv2d/kernel/Regularizer/Square:y:0(conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
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
identityIdentity:output:0*
_input_shapes
: 2b
/conv2d/kernel/Regularizer/Square/ReadVariableOp/conv2d/kernel/Regularizer/Square/ReadVariableOp
�
�
*__inference_CNN_2D_NORM_layer_call_fn_1702
input_1/
layer_normalization_gamma:x'.
layer_normalization_beta:x''
conv2d_kernel:		x
conv2d_bias:)
conv2d_1_kernel:
conv2d_1_bias:2
#batch_normalization_moving_variance:	�(
batch_normalization_gamma:	�.
batch_normalization_moving_mean:	�'
batch_normalization_beta:	�)
conv2d_2_kernel:
conv2d_2_bias:3
%batch_normalization_1_moving_variance:Z)
batch_normalization_1_gamma:Z/
!batch_normalization_1_moving_mean:Z(
batch_normalization_1_beta:Z3
%batch_normalization_2_moving_variance:Z)
batch_normalization_2_gamma:Z/
!batch_normalization_2_moving_mean:Z(
batch_normalization_2_beta:Z
dense_kernel:Z%

dense_bias:%
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1layer_normalization_gammalayer_normalization_betaconv2d_kernelconv2d_biasconv2d_1_kernelconv2d_1_bias#batch_normalization_moving_variancebatch_normalization_gammabatch_normalization_moving_meanbatch_normalization_betaconv2d_2_kernelconv2d_2_bias%batch_normalization_1_moving_variancebatch_normalization_1_gamma!batch_normalization_1_moving_meanbatch_normalization_1_beta%batch_normalization_2_moving_variancebatch_normalization_2_gamma!batch_normalization_2_moving_meanbatch_normalization_2_betadense_kernel
dense_bias*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������%*8
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_CNN_2D_NORM_layer_call_and_return_conditional_losses_1677o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������%`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*Z
_input_shapesI
G:���������x': : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������x'
!
_user_specified_name	input_1
�
d
H__inference_gaussian_noise_layer_call_and_return_conditional_losses_1459

inputs
identityV
IdentityIdentityinputs*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
��
�
E__inference_CNN_2D_NORM_layer_call_and_return_conditional_losses_2480
input_1C
-layer_normalization_layer_normalization_gamma:x'B
,layer_normalization_layer_normalization_beta:x'.
conv2d_conv2d_kernel:		x 
conv2d_conv2d_bias:2
conv2d_1_conv2d_1_kernel:$
conv2d_1_conv2d_1_bias:F
7batch_normalization_batch_normalization_moving_variance:	�<
-batch_normalization_batch_normalization_gamma:	�B
3batch_normalization_batch_normalization_moving_mean:	�;
,batch_normalization_batch_normalization_beta:	�2
conv2d_2_conv2d_2_kernel:$
conv2d_2_conv2d_2_bias:I
;batch_normalization_1_batch_normalization_1_moving_variance:Z?
1batch_normalization_1_batch_normalization_1_gamma:ZE
7batch_normalization_1_batch_normalization_1_moving_mean:Z>
0batch_normalization_1_batch_normalization_1_beta:ZI
;batch_normalization_2_batch_normalization_2_moving_variance:Z?
1batch_normalization_2_batch_normalization_2_gamma:ZE
7batch_normalization_2_batch_normalization_2_moving_mean:Z>
0batch_normalization_2_batch_normalization_2_beta:Z$
dense_dense_kernel:Z%
dense_dense_bias:%
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�-batch_normalization_2/StatefulPartitionedCall�conv2d/StatefulPartitionedCall�/conv2d/kernel/Regularizer/Square/ReadVariableOp� conv2d_1/StatefulPartitionedCall�1conv2d_1/kernel/Regularizer/Square/ReadVariableOp� conv2d_2/StatefulPartitionedCall�1conv2d_2/kernel/Regularizer/Square/ReadVariableOp�dense/StatefulPartitionedCall�.dense/kernel/Regularizer/Square/ReadVariableOp�+layer_normalization/StatefulPartitionedCall�
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCallinput_1-layer_normalization_layer_normalization_gamma,layer_normalization_layer_normalization_beta*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������x'*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *V
fQRO
M__inference_layer_normalization_layer_call_and_return_conditional_losses_1425�
conv2d/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0conv2d_conv2d_kernelconv2d_conv2d_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_1445�
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_1453�
gaussian_noise/PartitionedCallPartitionedCall&max_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_gaussian_noise_layer_call_and_return_conditional_losses_1459�
activation/PartitionedCallPartitionedCall'gaussian_noise/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_activation_layer_call_and_return_conditional_losses_1466�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0conv2d_1_conv2d_1_kernelconv2d_1_conv2d_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_1484�
flatten/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_1494�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:07batch_normalization_batch_normalization_moving_variance-batch_normalization_batch_normalization_gamma3batch_normalization_batch_normalization_moving_mean,batch_normalization_batch_normalization_beta*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_1019�
reshape/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_reshape_layer_call_and_return_conditional_losses_1515�
 gaussian_noise_1/PartitionedCallPartitionedCall reshape/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *S
fNRL
J__inference_gaussian_noise_1_layer_call_and_return_conditional_losses_1521�
activation_1/PartitionedCallPartitionedCall)gaussian_noise_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_activation_1_layer_call_and_return_conditional_losses_1528�
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0conv2d_2_conv2d_2_kernelconv2d_2_conv2d_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_1546�
flatten_1/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������Z* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_flatten_1_layer_call_and_return_conditional_losses_1556�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0;batch_normalization_1_batch_normalization_1_moving_variance1batch_normalization_1_batch_normalization_1_gamma7batch_normalization_1_batch_normalization_1_moving_mean0batch_normalization_1_batch_normalization_1_beta*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������Z*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1147�
reshape_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_reshape_1_layer_call_and_return_conditional_losses_1577�
 gaussian_noise_2/PartitionedCallPartitionedCall"reshape_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *S
fNRL
J__inference_gaussian_noise_2_layer_call_and_return_conditional_losses_1583�
activation_2/PartitionedCallPartitionedCall)gaussian_noise_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_activation_2_layer_call_and_return_conditional_losses_1590�
flatten_2/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������Z* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_flatten_2_layer_call_and_return_conditional_losses_1598�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0;batch_normalization_2_batch_normalization_2_moving_variance1batch_normalization_2_batch_normalization_2_gamma7batch_normalization_2_batch_normalization_2_moving_mean0batch_normalization_2_batch_normalization_2_beta*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������Z*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1275�
dense/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0dense_dense_kerneldense_dense_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������%*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_1621�
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
 *2
config_proto" 

CPU

GPU2 *0J 8� *4
f/R-
+__inference_dense_activity_regularizer_1636u
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
valueB:�
'dense/ActivityRegularizer/strided_sliceStridedSlice(dense/ActivityRegularizer/Shape:output:06dense/ActivityRegularizer/strided_slice/stack:output:08dense/ActivityRegularizer/strided_slice/stack_1:output:08dense/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
dense/ActivityRegularizer/CastCast0dense/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: �
!dense/ActivityRegularizer/truedivRealDiv2dense/ActivityRegularizer/PartitionedCall:output:0"dense/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: �
activation_3/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������%* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_activation_3_layer_call_and_return_conditional_losses_1650�
/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_conv2d_kernel*&
_output_shapes
:		x*
dtype0�
 conv2d/kernel/Regularizer/SquareSquare7conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		xx
conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d/kernel/Regularizer/SumSum$conv2d/kernel/Regularizer/Square:y:0(conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
conv2d/kernel/Regularizer/mulMul(conv2d/kernel/Regularizer/mul/x:output:0&conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_1_conv2d_1_kernel*&
_output_shapes
:*
dtype0�
"conv2d_1/kernel/Regularizer/SquareSquare9conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
!conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_1/kernel/Regularizer/SumSum&conv2d_1/kernel/Regularizer/Square:y:0*conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
conv2d_1/kernel/Regularizer/mulMul*conv2d_1/kernel/Regularizer/mul/x:output:0(conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_2_conv2d_2_kernel*&
_output_shapes
:*
dtype0�
"conv2d_2/kernel/Regularizer/SquareSquare9conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
!conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_2/kernel/Regularizer/SumSum&conv2d_2/kernel/Regularizer/Square:y:0*conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
conv2d_2/kernel/Regularizer/mulMul*conv2d_2/kernel/Regularizer/mul/x:output:0(conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_dense_kernel*
_output_shapes

:Z%*
dtype0�
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:Z%o
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: t
IdentityIdentity%activation_3/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������%�
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall^conv2d/StatefulPartitionedCall0^conv2d/kernel/Regularizer/Square/ReadVariableOp!^conv2d_1/StatefulPartitionedCall2^conv2d_1/kernel/Regularizer/Square/ReadVariableOp!^conv2d_2/StatefulPartitionedCall2^conv2d_2/kernel/Regularizer/Square/ReadVariableOp^dense/StatefulPartitionedCall/^dense/kernel/Regularizer/Square/ReadVariableOp,^layer_normalization/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*Z
_input_shapesI
G:���������x': : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2b
/conv2d/kernel/Regularizer/Square/ReadVariableOp/conv2d/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2f
1conv2d_1/kernel/Regularizer/Square/ReadVariableOp1conv2d_1/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2f
1conv2d_2/kernel/Regularizer/Square/ReadVariableOp1conv2d_2/kernel/Regularizer/Square/ReadVariableOp2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall:X T
/
_output_shapes
:���������x'
!
_user_specified_name	input_1
�
�
'__inference_conv2d_1_layer_call_fn_3189

inputs)
conv2d_1_kernel:
conv2d_1_bias:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_1_kernelconv2d_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_1484w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*2
_input_shapes!
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
D
(__inference_flatten_1_layer_call_fn_3376

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������Z* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_flatten_1_layer_call_and_return_conditional_losses_1556`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������Z"
identityIdentity:output:0*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
K
/__inference_gaussian_noise_2_layer_call_fn_3478

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *S
fNRL
J__inference_gaussian_noise_2_layer_call_and_return_conditional_losses_1583h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
_
C__inference_flatten_2_layer_call_and_return_conditional_losses_1598

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����Z   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:���������ZX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:���������Z"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
4__inference_batch_normalization_1_layer_call_fn_3391

inputs3
%batch_normalization_1_moving_variance:Z)
batch_normalization_1_gamma:Z/
!batch_normalization_1_moving_mean:Z(
batch_normalization_1_beta:Z
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs%batch_normalization_1_moving_variancebatch_normalization_1_gamma!batch_normalization_1_moving_meanbatch_normalization_1_beta*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������Z*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1147o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������Z`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*.
_input_shapes
:���������Z: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������Z
 
_user_specified_nameinputs
�
B
&__inference_reshape_layer_call_fn_3293

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_reshape_layer_call_and_return_conditional_losses_1515h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������	"
identityIdentity:output:0*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
f
J__inference_gaussian_noise_2_layer_call_and_return_conditional_losses_3487

inputs
identityV
IdentityIdentityinputs*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
f
-__inference_gaussian_noise_layer_call_fn_3151

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_gaussian_noise_layer_call_and_return_conditional_losses_1997w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*.
_input_shapes
:���������22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
C
input_18
serving_default_input_1:0���������x'@
activation_30
StatefulPartitionedCall:0���������%tensorflow/serving/predict:��
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
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
layer_with_weights-4
layer-12
layer-13
layer_with_weights-5
layer-14
layer-15
layer-16
layer-17
layer-18
layer_with_weights-6
layer-19
layer_with_weights-7
layer-20
layer-21
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
�__call__
+�&call_and_return_all_conditional_losses
�_default_save_signature"
_tf_keras_network
"
_tf_keras_input_layer
�
axis
	gamma
beta
 	variables
!trainable_variables
"regularization_losses
#	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

$kernel
%bias
&	variables
'trainable_variables
(regularization_losses
)	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
*	variables
+trainable_variables
,regularization_losses
-	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
.	variables
/trainable_variables
0regularization_losses
1	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
2	variables
3trainable_variables
4regularization_losses
5	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

6kernel
7bias
8	variables
9trainable_variables
:regularization_losses
;	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
<	variables
=trainable_variables
>regularization_losses
?	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
@axis
	Agamma
Bbeta
Cmoving_mean
Dmoving_variance
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

Ukernel
Vbias
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
[	variables
\trainable_variables
]regularization_losses
^	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
_axis
	`gamma
abeta
bmoving_mean
cmoving_variance
d	variables
etrainable_variables
fregularization_losses
g	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
h	variables
itrainable_variables
jregularization_losses
k	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
t	variables
utrainable_variables
vregularization_losses
w	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
xaxis
	ygamma
zbeta
{moving_mean
|moving_variance
}	variables
~trainable_variables
regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
"
	optimizer
�
0
1
$2
%3
64
75
A6
B7
C8
D9
U10
V11
`12
a13
b14
c15
y16
z17
{18
|19
�20
�21"
trackable_list_wrapper
�
0
1
$2
%3
64
75
A6
B7
U8
V9
`10
a11
y12
z13
�14
�15"
trackable_list_wrapper
@
�0
�1
�2
�3"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
 "
trackable_list_wrapper
/:-x'2layer_normalization/gamma
.:,x'2layer_normalization/beta
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
 	variables
!trainable_variables
"regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
':%		x2conv2d/kernel
:2conv2d/bias
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
&	variables
'trainable_variables
(regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
*	variables
+trainable_variables
,regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
.	variables
/trainable_variables
0regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
2	variables
3trainable_variables
4regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
):'2conv2d_1/kernel
:2conv2d_1/bias
.
60
71"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
8	variables
9trainable_variables
:regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
<	variables
=trainable_variables
>regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
(:&�2batch_normalization/gamma
':%�2batch_normalization/beta
0:.� (2batch_normalization/moving_mean
4:2� (2#batch_normalization/moving_variance
<
A0
B1
C2
D3"
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
):'2conv2d_2/kernel
:2conv2d_2/bias
.
U0
V1"
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
[	variables
\trainable_variables
]regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'Z2batch_normalization_1/gamma
(:&Z2batch_normalization_1/beta
1:/Z (2!batch_normalization_1/moving_mean
5:3Z (2%batch_normalization_1/moving_variance
<
`0
a1
b2
c3"
trackable_list_wrapper
.
`0
a1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
d	variables
etrainable_variables
fregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
h	variables
itrainable_variables
jregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
l	variables
mtrainable_variables
nregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
p	variables
qtrainable_variables
rregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
t	variables
utrainable_variables
vregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'Z2batch_normalization_2/gamma
(:&Z2batch_normalization_2/beta
1:/Z (2!batch_normalization_2/moving_mean
5:3Z (2%batch_normalization_2/moving_variance
<
y0
z1
{2
|3"
trackable_list_wrapper
.
y0
z1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
}	variables
~trainable_variables
regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:Z%2dense/kernel
:%2
dense/bias
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
�activity_regularizer_fn
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
J
C0
D1
b2
c3
{4
|5"
trackable_list_wrapper
�
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
21"
trackable_list_wrapper
8
�0
�1
�2"
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
�0"
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
�0"
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
C0
D1"
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
�0"
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
b0
c1"
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
.
{0
|1"
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
�0"
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
c

�total

�count
�
_fn_kwargs
�	variables
�	keras_api"
_tf_keras_metric
c

�total

�count
�
_fn_kwargs
�	variables
�	keras_api"
_tf_keras_metric
c

�total

�count
�
_fn_kwargs
�	variables
�	keras_api"
_tf_keras_metric
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total_1
:  (2count_1
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total_2
:  (2count_2
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
�2�
*__inference_CNN_2D_NORM_layer_call_fn_1702
*__inference_CNN_2D_NORM_layer_call_fn_2639
*__inference_CNN_2D_NORM_layer_call_fn_2666
*__inference_CNN_2D_NORM_layer_call_fn_2401�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
E__inference_CNN_2D_NORM_layer_call_and_return_conditional_losses_2831
E__inference_CNN_2D_NORM_layer_call_and_return_conditional_losses_3059
E__inference_CNN_2D_NORM_layer_call_and_return_conditional_losses_2480
E__inference_CNN_2D_NORM_layer_call_and_return_conditional_losses_2559�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
__inference__wrapped_model_978input_1"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
2__inference_layer_normalization_layer_call_fn_3066�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
M__inference_layer_normalization_layer_call_and_return_conditional_losses_3092�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
%__inference_conv2d_layer_call_fn_3105�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
@__inference_conv2d_layer_call_and_return_conditional_losses_3121�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
,__inference_max_pooling2d_layer_call_fn_3126
,__inference_max_pooling2d_layer_call_fn_3131�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_3136
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_3141�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
-__inference_gaussian_noise_layer_call_fn_3146
-__inference_gaussian_noise_layer_call_fn_3151�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
H__inference_gaussian_noise_layer_call_and_return_conditional_losses_3155
H__inference_gaussian_noise_layer_call_and_return_conditional_losses_3166�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
)__inference_activation_layer_call_fn_3171�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_activation_layer_call_and_return_conditional_losses_3176�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_conv2d_1_layer_call_fn_3189�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_conv2d_1_layer_call_and_return_conditional_losses_3205�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
&__inference_flatten_layer_call_fn_3210�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
A__inference_flatten_layer_call_and_return_conditional_losses_3216�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
2__inference_batch_normalization_layer_call_fn_3225
2__inference_batch_normalization_layer_call_fn_3234�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
M__inference_batch_normalization_layer_call_and_return_conditional_losses_3254
M__inference_batch_normalization_layer_call_and_return_conditional_losses_3288�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
&__inference_reshape_layer_call_fn_3293�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
A__inference_reshape_layer_call_and_return_conditional_losses_3307�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
/__inference_gaussian_noise_1_layer_call_fn_3312
/__inference_gaussian_noise_1_layer_call_fn_3317�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
J__inference_gaussian_noise_1_layer_call_and_return_conditional_losses_3321
J__inference_gaussian_noise_1_layer_call_and_return_conditional_losses_3332�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
+__inference_activation_1_layer_call_fn_3337�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
F__inference_activation_1_layer_call_and_return_conditional_losses_3342�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_conv2d_2_layer_call_fn_3355�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_conv2d_2_layer_call_and_return_conditional_losses_3371�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_flatten_1_layer_call_fn_3376�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_flatten_1_layer_call_and_return_conditional_losses_3382�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
4__inference_batch_normalization_1_layer_call_fn_3391
4__inference_batch_normalization_1_layer_call_fn_3400�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3420
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3454�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
(__inference_reshape_1_layer_call_fn_3459�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_reshape_1_layer_call_and_return_conditional_losses_3473�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
/__inference_gaussian_noise_2_layer_call_fn_3478
/__inference_gaussian_noise_2_layer_call_fn_3483�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
J__inference_gaussian_noise_2_layer_call_and_return_conditional_losses_3487
J__inference_gaussian_noise_2_layer_call_and_return_conditional_losses_3498�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
+__inference_activation_2_layer_call_fn_3503�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
F__inference_activation_2_layer_call_and_return_conditional_losses_3508�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_flatten_2_layer_call_fn_3513�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_flatten_2_layer_call_and_return_conditional_losses_3519�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
4__inference_batch_normalization_2_layer_call_fn_3528
4__inference_batch_normalization_2_layer_call_fn_3537�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3557
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3591�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
$__inference_dense_layer_call_fn_3604�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_dense_layer_call_and_return_all_conditional_losses_3613�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
+__inference_activation_3_layer_call_fn_3618�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
F__inference_activation_3_layer_call_and_return_conditional_losses_3623�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
__inference_loss_fn_0_3634�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_1_3645�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_2_3656�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_3_3667�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
"__inference_signature_wrapper_2612input_1"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
+__inference_dense_activity_regularizer_1392�
���
FullArgSpec
args�
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *�
	�
�2�
?__inference_dense_layer_call_and_return_conditional_losses_3683�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
E__inference_CNN_2D_NORM_layer_call_and_return_conditional_losses_2480�$%67DACBUVc`ba|y{z��@�=
6�3
)�&
input_1���������x'
p 

 
� "%�"
�
0���������%
� �
E__inference_CNN_2D_NORM_layer_call_and_return_conditional_losses_2559�$%67CDABUVbc`a{|yz��@�=
6�3
)�&
input_1���������x'
p

 
� "%�"
�
0���������%
� �
E__inference_CNN_2D_NORM_layer_call_and_return_conditional_losses_2831�$%67DACBUVc`ba|y{z��?�<
5�2
(�%
inputs���������x'
p 

 
� "%�"
�
0���������%
� �
E__inference_CNN_2D_NORM_layer_call_and_return_conditional_losses_3059�$%67CDABUVbc`a{|yz��?�<
5�2
(�%
inputs���������x'
p

 
� "%�"
�
0���������%
� �
*__inference_CNN_2D_NORM_layer_call_fn_1702v$%67DACBUVc`ba|y{z��@�=
6�3
)�&
input_1���������x'
p 

 
� "����������%�
*__inference_CNN_2D_NORM_layer_call_fn_2401v$%67CDABUVbc`a{|yz��@�=
6�3
)�&
input_1���������x'
p

 
� "����������%�
*__inference_CNN_2D_NORM_layer_call_fn_2639u$%67DACBUVc`ba|y{z��?�<
5�2
(�%
inputs���������x'
p 

 
� "����������%�
*__inference_CNN_2D_NORM_layer_call_fn_2666u$%67CDABUVbc`a{|yz��?�<
5�2
(�%
inputs���������x'
p

 
� "����������%�
__inference__wrapped_model_978�$%67DACBUVc`ba|y{z��8�5
.�+
)�&
input_1���������x'
� ";�8
6
activation_3&�#
activation_3���������%�
F__inference_activation_1_layer_call_and_return_conditional_losses_3342h7�4
-�*
(�%
inputs���������	
� "-�*
#� 
0���������	
� �
+__inference_activation_1_layer_call_fn_3337[7�4
-�*
(�%
inputs���������	
� " ����������	�
F__inference_activation_2_layer_call_and_return_conditional_losses_3508h7�4
-�*
(�%
inputs���������
� "-�*
#� 
0���������
� �
+__inference_activation_2_layer_call_fn_3503[7�4
-�*
(�%
inputs���������
� " �����������
F__inference_activation_3_layer_call_and_return_conditional_losses_3623X/�,
%�"
 �
inputs���������%
� "%�"
�
0���������%
� z
+__inference_activation_3_layer_call_fn_3618K/�,
%�"
 �
inputs���������%
� "����������%�
D__inference_activation_layer_call_and_return_conditional_losses_3176h7�4
-�*
(�%
inputs���������
� "-�*
#� 
0���������
� �
)__inference_activation_layer_call_fn_3171[7�4
-�*
(�%
inputs���������
� " �����������
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3420bc`ba3�0
)�&
 �
inputs���������Z
p 
� "%�"
�
0���������Z
� �
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3454bbc`a3�0
)�&
 �
inputs���������Z
p
� "%�"
�
0���������Z
� �
4__inference_batch_normalization_1_layer_call_fn_3391Uc`ba3�0
)�&
 �
inputs���������Z
p 
� "����������Z�
4__inference_batch_normalization_1_layer_call_fn_3400Ubc`a3�0
)�&
 �
inputs���������Z
p
� "����������Z�
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3557b|y{z3�0
)�&
 �
inputs���������Z
p 
� "%�"
�
0���������Z
� �
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3591b{|yz3�0
)�&
 �
inputs���������Z
p
� "%�"
�
0���������Z
� �
4__inference_batch_normalization_2_layer_call_fn_3528U|y{z3�0
)�&
 �
inputs���������Z
p 
� "����������Z�
4__inference_batch_normalization_2_layer_call_fn_3537U{|yz3�0
)�&
 �
inputs���������Z
p
� "����������Z�
M__inference_batch_normalization_layer_call_and_return_conditional_losses_3254dDACB4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
M__inference_batch_normalization_layer_call_and_return_conditional_losses_3288dCDAB4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
2__inference_batch_normalization_layer_call_fn_3225WDACB4�1
*�'
!�
inputs����������
p 
� "������������
2__inference_batch_normalization_layer_call_fn_3234WCDAB4�1
*�'
!�
inputs����������
p
� "������������
B__inference_conv2d_1_layer_call_and_return_conditional_losses_3205l677�4
-�*
(�%
inputs���������
� "-�*
#� 
0���������	
� �
'__inference_conv2d_1_layer_call_fn_3189_677�4
-�*
(�%
inputs���������
� " ����������	�
B__inference_conv2d_2_layer_call_and_return_conditional_losses_3371lUV7�4
-�*
(�%
inputs���������	
� "-�*
#� 
0���������
� �
'__inference_conv2d_2_layer_call_fn_3355_UV7�4
-�*
(�%
inputs���������	
� " �����������
@__inference_conv2d_layer_call_and_return_conditional_losses_3121l$%7�4
-�*
(�%
inputs���������x'
� "-�*
#� 
0���������
� �
%__inference_conv2d_layer_call_fn_3105_$%7�4
-�*
(�%
inputs���������x'
� " ����������U
+__inference_dense_activity_regularizer_1392&�
�
�	
x
� "� �
C__inference_dense_layer_call_and_return_all_conditional_losses_3613l��/�,
%�"
 �
inputs���������Z
� "3�0
�
0���������%
�
�	
1/0 �
?__inference_dense_layer_call_and_return_conditional_losses_3683^��/�,
%�"
 �
inputs���������Z
� "%�"
�
0���������%
� y
$__inference_dense_layer_call_fn_3604Q��/�,
%�"
 �
inputs���������Z
� "����������%�
C__inference_flatten_1_layer_call_and_return_conditional_losses_3382`7�4
-�*
(�%
inputs���������
� "%�"
�
0���������Z
� 
(__inference_flatten_1_layer_call_fn_3376S7�4
-�*
(�%
inputs���������
� "����������Z�
C__inference_flatten_2_layer_call_and_return_conditional_losses_3519`7�4
-�*
(�%
inputs���������
� "%�"
�
0���������Z
� 
(__inference_flatten_2_layer_call_fn_3513S7�4
-�*
(�%
inputs���������
� "����������Z�
A__inference_flatten_layer_call_and_return_conditional_losses_3216a7�4
-�*
(�%
inputs���������	
� "&�#
�
0����������
� ~
&__inference_flatten_layer_call_fn_3210T7�4
-�*
(�%
inputs���������	
� "������������
J__inference_gaussian_noise_1_layer_call_and_return_conditional_losses_3321l;�8
1�.
(�%
inputs���������	
p 
� "-�*
#� 
0���������	
� �
J__inference_gaussian_noise_1_layer_call_and_return_conditional_losses_3332l;�8
1�.
(�%
inputs���������	
p
� "-�*
#� 
0���������	
� �
/__inference_gaussian_noise_1_layer_call_fn_3312_;�8
1�.
(�%
inputs���������	
p 
� " ����������	�
/__inference_gaussian_noise_1_layer_call_fn_3317_;�8
1�.
(�%
inputs���������	
p
� " ����������	�
J__inference_gaussian_noise_2_layer_call_and_return_conditional_losses_3487l;�8
1�.
(�%
inputs���������
p 
� "-�*
#� 
0���������
� �
J__inference_gaussian_noise_2_layer_call_and_return_conditional_losses_3498l;�8
1�.
(�%
inputs���������
p
� "-�*
#� 
0���������
� �
/__inference_gaussian_noise_2_layer_call_fn_3478_;�8
1�.
(�%
inputs���������
p 
� " �����������
/__inference_gaussian_noise_2_layer_call_fn_3483_;�8
1�.
(�%
inputs���������
p
� " �����������
H__inference_gaussian_noise_layer_call_and_return_conditional_losses_3155l;�8
1�.
(�%
inputs���������
p 
� "-�*
#� 
0���������
� �
H__inference_gaussian_noise_layer_call_and_return_conditional_losses_3166l;�8
1�.
(�%
inputs���������
p
� "-�*
#� 
0���������
� �
-__inference_gaussian_noise_layer_call_fn_3146_;�8
1�.
(�%
inputs���������
p 
� " �����������
-__inference_gaussian_noise_layer_call_fn_3151_;�8
1�.
(�%
inputs���������
p
� " �����������
M__inference_layer_normalization_layer_call_and_return_conditional_losses_3092l7�4
-�*
(�%
inputs���������x'
� "-�*
#� 
0���������x'
� �
2__inference_layer_normalization_layer_call_fn_3066_7�4
-�*
(�%
inputs���������x'
� " ����������x'9
__inference_loss_fn_0_3634$�

� 
� "� 9
__inference_loss_fn_1_36456�

� 
� "� 9
__inference_loss_fn_2_3656U�

� 
� "� :
__inference_loss_fn_3_3667��

� 
� "� �
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_3136�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_3141h7�4
-�*
(�%
inputs���������
� "-�*
#� 
0���������
� �
,__inference_max_pooling2d_layer_call_fn_3126�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
,__inference_max_pooling2d_layer_call_fn_3131[7�4
-�*
(�%
inputs���������
� " �����������
C__inference_reshape_1_layer_call_and_return_conditional_losses_3473`/�,
%�"
 �
inputs���������Z
� "-�*
#� 
0���������
� 
(__inference_reshape_1_layer_call_fn_3459S/�,
%�"
 �
inputs���������Z
� " �����������
A__inference_reshape_layer_call_and_return_conditional_losses_3307a0�-
&�#
!�
inputs����������
� "-�*
#� 
0���������	
� ~
&__inference_reshape_layer_call_fn_3293T0�-
&�#
!�
inputs����������
� " ����������	�
"__inference_signature_wrapper_2612�$%67DACBUVc`ba|y{z��C�@
� 
9�6
4
input_1)�&
input_1���������x'";�8
6
activation_3&�#
activation_3���������%
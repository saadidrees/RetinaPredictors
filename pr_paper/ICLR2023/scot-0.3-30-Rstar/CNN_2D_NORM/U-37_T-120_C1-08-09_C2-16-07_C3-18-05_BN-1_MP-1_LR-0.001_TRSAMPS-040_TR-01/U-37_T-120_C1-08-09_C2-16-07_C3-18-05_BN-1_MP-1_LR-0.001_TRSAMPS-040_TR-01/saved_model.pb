цъ
ц
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
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
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

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

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

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
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	
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
dtypetype
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
list(type)(0
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
list(type)(0
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

2	
С
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
executor_typestring Ј
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
і
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

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.7.02v2.7.0-rc1-69-gc256c071bb28щ

layer_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:x'*,
shared_namelayer_normalization_1/gamma

/layer_normalization_1/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_1/gamma*"
_output_shapes
:x'*
dtype0

layer_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:x'*+
shared_namelayer_normalization_1/beta

.layer_normalization_1/beta/Read/ReadVariableOpReadVariableOplayer_normalization_1/beta*"
_output_shapes
:x'*
dtype0

conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:		x* 
shared_nameconv2d_6/kernel
{
#conv2d_6/kernel/Read/ReadVariableOpReadVariableOpconv2d_6/kernel*&
_output_shapes
:		x*
dtype0
r
conv2d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_6/bias
k
!conv2d_6/bias/Read/ReadVariableOpReadVariableOpconv2d_6/bias*
_output_shapes
:*
dtype0

conv2d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_7/kernel
{
#conv2d_7/kernel/Read/ReadVariableOpReadVariableOpconv2d_7/kernel*&
_output_shapes
:*
dtype0
r
conv2d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_7/bias
k
!conv2d_7/bias/Read/ReadVariableOpReadVariableOpconv2d_7/bias*
_output_shapes
:*
dtype0

batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:а**
shared_namebatch_normalization/gamma

-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes	
:а*
dtype0

batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:а*)
shared_namebatch_normalization/beta

,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes	
:а*
dtype0

batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:а*0
shared_name!batch_normalization/moving_mean

3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes	
:а*
dtype0

#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:а*4
shared_name%#batch_normalization/moving_variance

7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes	
:а*
dtype0

conv2d_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_8/kernel
{
#conv2d_8/kernel/Read/ReadVariableOpReadVariableOpconv2d_8/kernel*&
_output_shapes
:*
dtype0
r
conv2d_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_8/bias
k
!conv2d_8/bias/Read/ReadVariableOpReadVariableOpconv2d_8/bias*
_output_shapes
:*
dtype0

batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*,
shared_namebatch_normalization_1/gamma

/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes
:Z*
dtype0

batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*+
shared_namebatch_normalization_1/beta

.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes
:Z*
dtype0

!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*2
shared_name#!batch_normalization_1/moving_mean

5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes
:Z*
dtype0
Ђ
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*6
shared_name'%batch_normalization_1/moving_variance

9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes
:Z*
dtype0

batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*,
shared_namebatch_normalization_2/gamma

/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes
:Z*
dtype0

batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*+
shared_namebatch_normalization_2/beta

.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes
:Z*
dtype0

!batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*2
shared_name#!batch_normalization_2/moving_mean

5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes
:Z*
dtype0
Ђ
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*6
shared_name'%batch_normalization_2/moving_variance

9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
_output_shapes
:Z*
dtype0
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z%*
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

:Z%*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:%*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:%*
dtype0
d
total_12VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
total_12
]
total_12/Read/ReadVariableOpReadVariableOptotal_12*
_output_shapes
: *
dtype0
d
count_12VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
count_12
]
count_12/Read/ReadVariableOpReadVariableOpcount_12*
_output_shapes
: *
dtype0
d
total_13VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
total_13
]
total_13/Read/ReadVariableOpReadVariableOptotal_13*
_output_shapes
: *
dtype0
d
count_13VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
count_13
]
count_13/Read/ReadVariableOpReadVariableOpcount_13*
_output_shapes
: *
dtype0
d
total_14VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
total_14
]
total_14/Read/ReadVariableOpReadVariableOptotal_14*
_output_shapes
: *
dtype0
d
count_14VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
count_14
]
count_14/Read/ReadVariableOpReadVariableOpcount_14*
_output_shapes
: *
dtype0

NoOpNoOp
фV
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*V
valueVBV BV
ы
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

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

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

xaxis
	ygamma
zbeta
{moving_mean
|moving_variance
}	variables
~trainable_variables
regularization_losses
	keras_api
n
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
V
	variables
trainable_variables
regularization_losses
	keras_api
 
Ј
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
20
21
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
14
15
 
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
 
 
fd
VARIABLE_VALUElayer_normalization_1/gamma5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUElayer_normalization_1/beta4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
 	variables
!trainable_variables
"regularization_losses
[Y
VARIABLE_VALUEconv2d_6/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_6/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

$0
%1

$0
%1
 
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
&	variables
'trainable_variables
(regularization_losses
 
 
 
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
*	variables
+trainable_variables
,regularization_losses
 
 
 
В
non_trainable_variables
 layers
Ёmetrics
 Ђlayer_regularization_losses
Ѓlayer_metrics
.	variables
/trainable_variables
0regularization_losses
 
 
 
В
Єnon_trainable_variables
Ѕlayers
Іmetrics
 Їlayer_regularization_losses
Јlayer_metrics
2	variables
3trainable_variables
4regularization_losses
[Y
VARIABLE_VALUEconv2d_7/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_7/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

60
71

60
71
 
В
Љnon_trainable_variables
Њlayers
Ћmetrics
 Ќlayer_regularization_losses
­layer_metrics
8	variables
9trainable_variables
:regularization_losses
 
 
 
В
Ўnon_trainable_variables
Џlayers
Аmetrics
 Бlayer_regularization_losses
Вlayer_metrics
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
В
Гnon_trainable_variables
Дlayers
Еmetrics
 Жlayer_regularization_losses
Зlayer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
 
 
 
В
Иnon_trainable_variables
Йlayers
Кmetrics
 Лlayer_regularization_losses
Мlayer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
 
 
 
В
Нnon_trainable_variables
Оlayers
Пmetrics
 Рlayer_regularization_losses
Сlayer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
 
 
 
В
Тnon_trainable_variables
Уlayers
Фmetrics
 Хlayer_regularization_losses
Цlayer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
[Y
VARIABLE_VALUEconv2d_8/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_8/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

U0
V1

U0
V1
 
В
Чnon_trainable_variables
Шlayers
Щmetrics
 Ъlayer_regularization_losses
Ыlayer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
 
 
 
В
Ьnon_trainable_variables
Эlayers
Юmetrics
 Яlayer_regularization_losses
аlayer_metrics
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
В
бnon_trainable_variables
вlayers
гmetrics
 дlayer_regularization_losses
еlayer_metrics
d	variables
etrainable_variables
fregularization_losses
 
 
 
В
жnon_trainable_variables
зlayers
иmetrics
 йlayer_regularization_losses
кlayer_metrics
h	variables
itrainable_variables
jregularization_losses
 
 
 
В
лnon_trainable_variables
мlayers
нmetrics
 оlayer_regularization_losses
пlayer_metrics
l	variables
mtrainable_variables
nregularization_losses
 
 
 
В
рnon_trainable_variables
сlayers
тmetrics
 уlayer_regularization_losses
фlayer_metrics
p	variables
qtrainable_variables
rregularization_losses
 
 
 
В
хnon_trainable_variables
цlayers
чmetrics
 шlayer_regularization_losses
щlayer_metrics
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
В
ъnon_trainable_variables
ыlayers
ьmetrics
 эlayer_regularization_losses
юlayer_metrics
}	variables
~trainable_variables
regularization_losses
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
Е
яnon_trainable_variables
№layers
ёmetrics
 ђlayer_regularization_losses
ѓlayer_metrics
	variables
trainable_variables
regularization_losses
 
 
 
Е
єnon_trainable_variables
ѕlayers
іmetrics
 їlayer_regularization_losses
јlayer_metrics
	variables
trainable_variables
regularization_losses
*
C0
D1
b2
c3
{4
|5
І
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
љ0
њ1
ћ2
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

ќtotal

§count
ў
_fn_kwargs
џ	variables
	keras_api
I

total

count

_fn_kwargs
	variables
	keras_api
I

total

count

_fn_kwargs
	variables
	keras_api
RP
VARIABLE_VALUEtotal_124keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEcount_124keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 

ќ0
§1

џ	variables
RP
VARIABLE_VALUEtotal_134keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEcount_134keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

	variables
RP
VARIABLE_VALUEtotal_144keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEcount_144keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

	variables

serving_default_input_4Placeholder*/
_output_shapes
:џџџџџџџџџx'*
dtype0*$
shape:џџџџџџџџџx'

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_4layer_normalization_1/gammalayer_normalization_1/betaconv2d_6/kernelconv2d_6/biasconv2d_7/kernelconv2d_7/bias#batch_normalization/moving_variancebatch_normalization/gammabatch_normalization/moving_meanbatch_normalization/betaconv2d_8/kernelconv2d_8/bias%batch_normalization_1/moving_variancebatch_normalization_1/gamma!batch_normalization_1/moving_meanbatch_normalization_1/beta%batch_normalization_2/moving_variancebatch_normalization_2/gamma!batch_normalization_2/moving_meanbatch_normalization_2/betadense_2/kerneldense_2/bias*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ%*8
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 *+
f&R$
"__inference_signature_wrapper_9602
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
п
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename/layer_normalization_1/gamma/Read/ReadVariableOp.layer_normalization_1/beta/Read/ReadVariableOp#conv2d_6/kernel/Read/ReadVariableOp!conv2d_6/bias/Read/ReadVariableOp#conv2d_7/kernel/Read/ReadVariableOp!conv2d_7/bias/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp#conv2d_8/kernel/Read/ReadVariableOp!conv2d_8/bias/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp/batch_normalization_2/gamma/Read/ReadVariableOp.batch_normalization_2/beta/Read/ReadVariableOp5batch_normalization_2/moving_mean/Read/ReadVariableOp9batch_normalization_2/moving_variance/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOptotal_12/Read/ReadVariableOpcount_12/Read/ReadVariableOptotal_13/Read/ReadVariableOpcount_13/Read/ReadVariableOptotal_14/Read/ReadVariableOpcount_14/Read/ReadVariableOpConst*)
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
GPU2 *0J 8 *'
f"R 
__inference__traced_save_10780
Њ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamelayer_normalization_1/gammalayer_normalization_1/betaconv2d_6/kernelconv2d_6/biasconv2d_7/kernelconv2d_7/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv2d_8/kernelconv2d_8/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_variancebatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_variancedense_2/kerneldense_2/biastotal_12count_12total_13count_13total_14count_14*(
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
GPU2 *0J 8 **
f%R#
!__inference__traced_restore_10874ОН

E
)__inference_flatten_2_layer_call_fn_10200

inputs
identityД
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџа* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_flatten_2_layer_call_and_return_conditional_losses_8484a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџа"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ	:W S
/
_output_shapes
:џџџџџџџџџ	
 
_user_specified_nameinputs

f
J__inference_gaussian_noise_6_layer_call_and_return_conditional_losses_8449

inputs
identityV
IdentityIdentityinputs*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
У
_
C__inference_flatten_3_layer_call_and_return_conditional_losses_8546

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџZ   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:џџџџџџџџџZX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџZ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ѕ
L
0__inference_gaussian_noise_6_layer_call_fn_10136

inputs
identityТ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_gaussian_noise_6_layer_call_and_return_conditional_losses_8449h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Є

E__inference_CNN_2D_NORM_layer_call_and_return_conditional_losses_9181

inputsG
1layer_normalization_1_layer_normalization_1_gamma:x'F
0layer_normalization_1_layer_normalization_1_beta:x'2
conv2d_6_conv2d_6_kernel:		x$
conv2d_6_conv2d_6_bias:2
conv2d_7_conv2d_7_kernel:$
conv2d_7_conv2d_7_bias:B
3batch_normalization_batch_normalization_moving_mean:	аF
7batch_normalization_batch_normalization_moving_variance:	а<
-batch_normalization_batch_normalization_gamma:	а;
,batch_normalization_batch_normalization_beta:	а2
conv2d_8_conv2d_8_kernel:$
conv2d_8_conv2d_8_bias:E
7batch_normalization_1_batch_normalization_1_moving_mean:ZI
;batch_normalization_1_batch_normalization_1_moving_variance:Z?
1batch_normalization_1_batch_normalization_1_gamma:Z>
0batch_normalization_1_batch_normalization_1_beta:ZE
7batch_normalization_2_batch_normalization_2_moving_mean:ZI
;batch_normalization_2_batch_normalization_2_moving_variance:Z?
1batch_normalization_2_batch_normalization_2_gamma:Z>
0batch_normalization_2_batch_normalization_2_beta:Z(
dense_2_dense_2_kernel:Z%"
dense_2_dense_2_bias:%
identityЂ+batch_normalization/StatefulPartitionedCallЂ-batch_normalization_1/StatefulPartitionedCallЂ-batch_normalization_2/StatefulPartitionedCallЂ conv2d_6/StatefulPartitionedCallЂ1conv2d_6/kernel/Regularizer/Square/ReadVariableOpЂ conv2d_7/StatefulPartitionedCallЂ1conv2d_7/kernel/Regularizer/Square/ReadVariableOpЂ conv2d_8/StatefulPartitionedCallЂ1conv2d_8/kernel/Regularizer/Square/ReadVariableOpЂdense_2/StatefulPartitionedCallЂ0dense_2/kernel/Regularizer/Square/ReadVariableOpЂ(gaussian_noise_6/StatefulPartitionedCallЂ(gaussian_noise_7/StatefulPartitionedCallЂ(gaussian_noise_8/StatefulPartitionedCallЂ-layer_normalization_1/StatefulPartitionedCallи
-layer_normalization_1/StatefulPartitionedCallStatefulPartitionedCallinputs1layer_normalization_1_layer_normalization_1_gamma0layer_normalization_1_layer_normalization_1_beta*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџx'*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *X
fSRQ
O__inference_layer_normalization_1_layer_call_and_return_conditional_losses_8415Л
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_1/StatefulPartitionedCall:output:0conv2d_6_conv2d_6_kernelconv2d_6_conv2d_6_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_conv2d_6_layer_call_and_return_conditional_losses_8435є
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_8443
(gaussian_noise_6/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_gaussian_noise_6_layer_call_and_return_conditional_losses_8987і
activation_8/PartitionedCallPartitionedCall1gaussian_noise_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_activation_8_layer_call_and_return_conditional_losses_8456Њ
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall%activation_8/PartitionedCall:output:0conv2d_7_conv2d_7_kernelconv2d_7_conv2d_7_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ	*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_conv2d_7_layer_call_and_return_conditional_losses_8474с
flatten_2/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџа* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_flatten_2_layer_call_and_return_conditional_losses_8484б
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:03batch_normalization_batch_normalization_moving_mean7batch_normalization_batch_normalization_moving_variance-batch_normalization_batch_normalization_gamma,batch_normalization_batch_normalization_beta*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџа*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_8052я
reshape/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ	* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_reshape_layer_call_and_return_conditional_losses_8505Ј
(gaussian_noise_7/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0)^gaussian_noise_6/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ	* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_gaussian_noise_7_layer_call_and_return_conditional_losses_8883і
activation_9/PartitionedCallPartitionedCall1gaussian_noise_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ	* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_activation_9_layer_call_and_return_conditional_losses_8518Њ
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall%activation_9/PartitionedCall:output:0conv2d_8_conv2d_8_kernelconv2d_8_conv2d_8_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_conv2d_8_layer_call_and_return_conditional_losses_8536р
flatten_3/PartitionedCallPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџZ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_flatten_3_layer_call_and_return_conditional_losses_8546ф
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:07batch_normalization_1_batch_normalization_1_moving_mean;batch_normalization_1_batch_normalization_1_moving_variance1batch_normalization_1_batch_normalization_1_gamma0batch_normalization_1_batch_normalization_1_beta*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџZ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_8180ѕ
reshape_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_reshape_1_layer_call_and_return_conditional_losses_8567Њ
(gaussian_noise_8/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0)^gaussian_noise_7/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_gaussian_noise_8_layer_call_and_return_conditional_losses_8779ј
activation_10/PartitionedCallPartitionedCall1gaussian_noise_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_activation_10_layer_call_and_return_conditional_losses_8580н
flatten_4/PartitionedCallPartitionedCall&activation_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџZ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_flatten_4_layer_call_and_return_conditional_losses_8588ф
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:07batch_normalization_2_batch_normalization_2_moving_mean;batch_normalization_2_batch_normalization_2_moving_variance1batch_normalization_2_batch_normalization_2_gamma0batch_normalization_2_batch_normalization_2_beta*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџZ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_8308­
dense_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0dense_2_dense_2_kerneldense_2_dense_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ%*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_8611Ъ
+dense_2/ActivityRegularizer/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8 *6
f1R/
-__inference_dense_2_activity_regularizer_8626y
!dense_2/ActivityRegularizer/ShapeShape(dense_2/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:y
/dense_2/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1dense_2/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1dense_2/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:н
)dense_2/ActivityRegularizer/strided_sliceStridedSlice*dense_2/ActivityRegularizer/Shape:output:08dense_2/ActivityRegularizer/strided_slice/stack:output:0:dense_2/ActivityRegularizer/strided_slice/stack_1:output:0:dense_2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
 dense_2/ActivityRegularizer/CastCast2dense_2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: Ћ
#dense_2/ActivityRegularizer/truedivRealDiv4dense_2/ActivityRegularizer/PartitionedCall:output:0$dense_2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ч
activation_11/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ%* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_activation_11_layer_call_and_return_conditional_losses_8640
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_6_conv2d_6_kernel*&
_output_shapes
:		x*
dtype0
"conv2d_6/kernel/Regularizer/SquareSquare9conv2d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		xz
!conv2d_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
conv2d_6/kernel/Regularizer/SumSum&conv2d_6/kernel/Regularizer/Square:y:0*conv2d_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0(conv2d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
1conv2d_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_7_conv2d_7_kernel*&
_output_shapes
:*
dtype0
"conv2d_7/kernel/Regularizer/SquareSquare9conv2d_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
!conv2d_7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
conv2d_7/kernel/Regularizer/SumSum&conv2d_7/kernel/Regularizer/Square:y:0*conv2d_7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv2d_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:
conv2d_7/kernel/Regularizer/mulMul*conv2d_7/kernel/Regularizer/mul/x:output:0(conv2d_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
1conv2d_8/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_8_conv2d_8_kernel*&
_output_shapes
:*
dtype0
"conv2d_8/kernel/Regularizer/SquareSquare9conv2d_8/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
!conv2d_8/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
conv2d_8/kernel/Regularizer/SumSum&conv2d_8/kernel/Regularizer/Square:y:0*conv2d_8/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv2d_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:
conv2d_8/kernel/Regularizer/mulMul*conv2d_8/kernel/Regularizer/mul/x:output:0(conv2d_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_2_dense_2_kernel*
_output_shapes

:Z%*
dtype0
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:Z%q
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_2/kernel/Regularizer/SumSum%dense_2/kernel/Regularizer/Square:y:0)dense_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: u
IdentityIdentity&activation_11/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ%п
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall2^conv2d_6/kernel/Regularizer/Square/ReadVariableOp!^conv2d_7/StatefulPartitionedCall2^conv2d_7/kernel/Regularizer/Square/ReadVariableOp!^conv2d_8/StatefulPartitionedCall2^conv2d_8/kernel/Regularizer/Square/ReadVariableOp ^dense_2/StatefulPartitionedCall1^dense_2/kernel/Regularizer/Square/ReadVariableOp)^gaussian_noise_6/StatefulPartitionedCall)^gaussian_noise_7/StatefulPartitionedCall)^gaussian_noise_8/StatefulPartitionedCall.^layer_normalization_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:џџџџџџџџџx': : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2f
1conv2d_6/kernel/Regularizer/Square/ReadVariableOp1conv2d_6/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2f
1conv2d_7/kernel/Regularizer/Square/ReadVariableOp1conv2d_7/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2f
1conv2d_8/kernel/Regularizer/Square/ReadVariableOp1conv2d_8/kernel/Regularizer/Square/ReadVariableOp2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2d
0dense_2/kernel/Regularizer/Square/ReadVariableOp0dense_2/kernel/Regularizer/Square/ReadVariableOp2T
(gaussian_noise_6/StatefulPartitionedCall(gaussian_noise_6/StatefulPartitionedCall2T
(gaussian_noise_7/StatefulPartitionedCall(gaussian_noise_7/StatefulPartitionedCall2T
(gaussian_noise_8/StatefulPartitionedCall(gaussian_noise_8/StatefulPartitionedCall2^
-layer_normalization_1/StatefulPartitionedCall-layer_normalization_1/StatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџx'
 
_user_specified_nameinputs

C
'__inference_reshape_layer_call_fn_10283

inputs
identityЙ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ	* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_reshape_layer_call_and_return_conditional_losses_8505h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ	"
identityIdentity:output:0*'
_input_shapes
:џџџџџџџџџа:P L
(
_output_shapes
:џџџџџџџџџа
 
_user_specified_nameinputs
а
D
-__inference_dense_2_activity_regularizer_8382
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
:џџџџџџџџџD
SumSumAbs:y:0range:output:0*
T0*
_output_shapes
: J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:I
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
Х
_
C__inference_flatten_2_layer_call_and_return_conditional_losses_8484

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџа  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџаY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџа"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ	:W S
/
_output_shapes
:џџџџџџџџџ	
 
_user_specified_nameinputs
С
c
G__inference_activation_8_layer_call_and_return_conditional_losses_10166

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:џџџџџџџџџb
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


M__inference_batch_normalization_layer_call_and_return_conditional_losses_8009

inputsK
<batchnorm_readvariableop_batch_normalization_moving_variance:	аE
6batchnorm_mul_readvariableop_batch_normalization_gamma:	аI
:batchnorm_readvariableop_1_batch_normalization_moving_mean:	аB
3batchnorm_readvariableop_2_batch_normalization_beta:	а
identityЂbatchnorm/ReadVariableOpЂbatchnorm/ReadVariableOp_1Ђbatchnorm/ReadVariableOp_2Ђbatchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp<batchnorm_readvariableop_batch_normalization_moving_variance*
_output_shapes	
:а*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:аQ
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:а
batchnorm/mul/ReadVariableOpReadVariableOp6batchnorm_mul_readvariableop_batch_normalization_gamma*
_output_shapes	
:а*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:аd
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:џџџџџџџџџа
batchnorm/ReadVariableOp_1ReadVariableOp:batchnorm_readvariableop_1_batch_normalization_moving_mean*
_output_shapes	
:а*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:а
batchnorm/ReadVariableOp_2ReadVariableOp3batchnorm_readvariableop_2_batch_normalization_beta*
_output_shapes	
:а*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:аs
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:џџџџџџџџџаc
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџаК
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџа: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџа
 
_user_specified_nameinputs
І	
i
J__inference_gaussian_noise_6_layer_call_and_return_conditional_losses_8987

inputs
identity;
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
 *ЭЬЬ=
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
dtype0
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*/
_output_shapes
:џџџџџџџџџa
addAddV2inputsrandom_normal:z:0*
T0*/
_output_shapes
:џџџџџџџџџW
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_8265

inputsL
>batchnorm_readvariableop_batch_normalization_2_moving_variance:ZF
8batchnorm_mul_readvariableop_batch_normalization_2_gamma:ZJ
<batchnorm_readvariableop_1_batch_normalization_2_moving_mean:ZC
5batchnorm_readvariableop_2_batch_normalization_2_beta:Z
identityЂbatchnorm/ReadVariableOpЂbatchnorm/ReadVariableOp_1Ђbatchnorm/ReadVariableOp_2Ђbatchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp>batchnorm_readvariableop_batch_normalization_2_moving_variance*
_output_shapes
:Z*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:ZP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:Z
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
:џџџџџџџџџZ
batchnorm/ReadVariableOp_1ReadVariableOp<batchnorm_readvariableop_1_batch_normalization_2_moving_mean*
_output_shapes
:Z*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:Z
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
:џџџџџџџџџZb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџZК
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџZ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџZ
 
_user_specified_nameinputs
С
c
G__inference_activation_9_layer_call_and_return_conditional_losses_10332

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:џџџџџџџџџ	b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ	"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ	:W S
/
_output_shapes
:џџџџџџџџџ	
 
_user_specified_nameinputs

И
P__inference_layer_normalization_1_layer_call_and_return_conditional_losses_10082

inputsH
2reshape_readvariableop_layer_normalization_1_gamma:x'I
3reshape_1_readvariableop_layer_normalization_1_beta:x'
identityЂReshape/ReadVariableOpЂReshape_1/ReadVariableOps
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
	keep_dims(u
moments/StopGradientStopGradientmoments/mean:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*/
_output_shapes
:џџџџџџџџџx'w
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         Џ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
	keep_dims(
Reshape/ReadVariableOpReadVariableOp2reshape_readvariableop_layer_normalization_1_gamma*"
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
:x'
Reshape_1/ReadVariableOpReadVariableOp3reshape_1_readvariableop_layer_normalization_1_beta*"
_output_shapes
:x'*
dtype0h
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   x      '   
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:x'T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Пж3
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџe
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*/
_output_shapes
:џџџџџџџџџu
batchnorm/mulMulbatchnorm/Rsqrt:y:0Reshape:output:0*
T0*/
_output_shapes
:џџџџџџџџџx'k
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*/
_output_shapes
:џџџџџџџџџx'z
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*/
_output_shapes
:џџџџџџџџџx'w
batchnorm/subSubReshape_1:output:0batchnorm/mul_2:z:0*
T0*/
_output_shapes
:џџџџџџџџџx'z
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*/
_output_shapes
:џџџџџџџџџx'j
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџx'z
NoOpNoOp^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџx': : 20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџx'
 
_user_specified_nameinputs
ы
c
G__inference_activation_10_layer_call_and_return_conditional_losses_8580

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:џџџџџџџџџb
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
З
А
B__inference_dense_2_layer_call_and_return_conditional_losses_10673

inputs6
$matmul_readvariableop_dense_2_kernel:Z%1
#biasadd_readvariableop_dense_2_bias:%
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂ0dense_2/kernel/Regularizer/Square/ReadVariableOpz
MatMul/ReadVariableOpReadVariableOp$matmul_readvariableop_dense_2_kernel*
_output_shapes

:Z%*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ%v
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_2_bias*
_output_shapes
:%*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ%
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp$matmul_readvariableop_dense_2_kernel*
_output_shapes

:Z%*
dtype0
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:Z%q
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_2/kernel/Regularizer/SumSum%dense_2/kernel/Regularizer/Square:y:0)dense_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ%Њ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp1^dense_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџZ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2d
0dense_2/kernel/Regularizer/Square/ReadVariableOp0dense_2/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџZ
 
_user_specified_nameinputs
ћ=
У
__inference__traced_save_10780
file_prefix:
6savev2_layer_normalization_1_gamma_read_readvariableop9
5savev2_layer_normalization_1_beta_read_readvariableop.
*savev2_conv2d_6_kernel_read_readvariableop,
(savev2_conv2d_6_bias_read_readvariableop.
*savev2_conv2d_7_kernel_read_readvariableop,
(savev2_conv2d_7_bias_read_readvariableop8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop.
*savev2_conv2d_8_kernel_read_readvariableop,
(savev2_conv2d_8_bias_read_readvariableop:
6savev2_batch_normalization_1_gamma_read_readvariableop9
5savev2_batch_normalization_1_beta_read_readvariableop@
<savev2_batch_normalization_1_moving_mean_read_readvariableopD
@savev2_batch_normalization_1_moving_variance_read_readvariableop:
6savev2_batch_normalization_2_gamma_read_readvariableop9
5savev2_batch_normalization_2_beta_read_readvariableop@
<savev2_batch_normalization_2_moving_mean_read_readvariableopD
@savev2_batch_normalization_2_moving_variance_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop'
#savev2_total_12_read_readvariableop'
#savev2_count_12_read_readvariableop'
#savev2_total_13_read_readvariableop'
#savev2_count_13_read_readvariableop'
#savev2_total_14_read_readvariableop'
#savev2_count_14_read_readvariableop
savev2_const

identity_1ЂMergeV2Checkpointsw
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
_temp/part
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
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Д
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*н
valueгBаB5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЇ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*M
valueDBBB B B B B B B B B B B B B B B B B B B B B B B B B B B B B Д
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:06savev2_layer_normalization_1_gamma_read_readvariableop5savev2_layer_normalization_1_beta_read_readvariableop*savev2_conv2d_6_kernel_read_readvariableop(savev2_conv2d_6_bias_read_readvariableop*savev2_conv2d_7_kernel_read_readvariableop(savev2_conv2d_7_bias_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop*savev2_conv2d_8_kernel_read_readvariableop(savev2_conv2d_8_bias_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop6savev2_batch_normalization_2_gamma_read_readvariableop5savev2_batch_normalization_2_beta_read_readvariableop<savev2_batch_normalization_2_moving_mean_read_readvariableop@savev2_batch_normalization_2_moving_variance_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop#savev2_total_12_read_readvariableop#savev2_count_12_read_readvariableop#savev2_total_13_read_readvariableop#savev2_count_13_read_readvariableop#savev2_total_14_read_readvariableop#savev2_count_14_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *+
dtypes!
2
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
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

identity_1Identity_1:output:0*х
_input_shapesг
а: :x':x':		x::::а:а:а:а:::Z:Z:Z:Z:Z:Z:Z:Z:Z%:%: : : : : : : 2(
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
:а:!

_output_shapes	
:а:!	

_output_shapes	
:а:!


_output_shapes	
:а:,(
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
њ
D
-__inference_dense_2_activity_regularizer_8626
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
:џџџџџџџџџD
SumSumAbs:y:0range:output:0*
T0*
_output_shapes
: J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:I
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

f
J__inference_gaussian_noise_7_layer_call_and_return_conditional_losses_8511

inputs
identityV
IdentityIdentityinputs*
T0*/
_output_shapes
:џџџџџџџџџ	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ	:W S
/
_output_shapes
:џџџџџџџџџ	
 
_user_specified_nameinputs
Ъ
_
C__inference_reshape_1_layer_call_and_return_conditional_losses_8567

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
valueB:б
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
value	B :Љ
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ`
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџZ:O K
'
_output_shapes
:џџџџџџџџџZ
 
_user_specified_nameinputs
ъ
b
F__inference_activation_9_layer_call_and_return_conditional_losses_8518

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:џџџџџџџџџ	b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ	:W S
/
_output_shapes
:џџџџџџџџџ	
 
_user_specified_nameinputs
к
Љ
(__inference_conv2d_6_layer_call_fn_10095

inputs)
conv2d_6_kernel:		x
conv2d_6_bias:
identityЂStatefulPartitionedCall№
StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_6_kernelconv2d_6_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_conv2d_6_layer_call_and_return_conditional_losses_8435w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџx': : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџx'
 
_user_specified_nameinputs
Ѓ
K
/__inference_max_pooling2d_2_layer_call_fn_10121

inputs
identityС
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_8443h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
А	
Љ
5__inference_batch_normalization_1_layer_call_fn_10390

inputs/
!batch_normalization_1_moving_mean:Z3
%batch_normalization_1_moving_variance:Z)
batch_normalization_1_gamma:Z(
batch_normalization_1_beta:Z
identityЂStatefulPartitionedCallк
StatefulPartitionedCallStatefulPartitionedCallinputs!batch_normalization_1_moving_mean%batch_normalization_1_moving_variancebatch_normalization_1_gammabatch_normalization_1_beta*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџZ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_8180o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџZ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџZ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџZ
 
_user_specified_nameinputs
Ј
e
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_7977

inputs
identityЙ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
data_formatNCHW*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
§

E__inference_CNN_2D_NORM_layer_call_and_return_conditional_losses_9549
input_4G
1layer_normalization_1_layer_normalization_1_gamma:x'F
0layer_normalization_1_layer_normalization_1_beta:x'2
conv2d_6_conv2d_6_kernel:		x$
conv2d_6_conv2d_6_bias:2
conv2d_7_conv2d_7_kernel:$
conv2d_7_conv2d_7_bias:B
3batch_normalization_batch_normalization_moving_mean:	аF
7batch_normalization_batch_normalization_moving_variance:	а<
-batch_normalization_batch_normalization_gamma:	а;
,batch_normalization_batch_normalization_beta:	а2
conv2d_8_conv2d_8_kernel:$
conv2d_8_conv2d_8_bias:E
7batch_normalization_1_batch_normalization_1_moving_mean:ZI
;batch_normalization_1_batch_normalization_1_moving_variance:Z?
1batch_normalization_1_batch_normalization_1_gamma:Z>
0batch_normalization_1_batch_normalization_1_beta:ZE
7batch_normalization_2_batch_normalization_2_moving_mean:ZI
;batch_normalization_2_batch_normalization_2_moving_variance:Z?
1batch_normalization_2_batch_normalization_2_gamma:Z>
0batch_normalization_2_batch_normalization_2_beta:Z(
dense_2_dense_2_kernel:Z%"
dense_2_dense_2_bias:%
identityЂ+batch_normalization/StatefulPartitionedCallЂ-batch_normalization_1/StatefulPartitionedCallЂ-batch_normalization_2/StatefulPartitionedCallЂ conv2d_6/StatefulPartitionedCallЂ1conv2d_6/kernel/Regularizer/Square/ReadVariableOpЂ conv2d_7/StatefulPartitionedCallЂ1conv2d_7/kernel/Regularizer/Square/ReadVariableOpЂ conv2d_8/StatefulPartitionedCallЂ1conv2d_8/kernel/Regularizer/Square/ReadVariableOpЂdense_2/StatefulPartitionedCallЂ0dense_2/kernel/Regularizer/Square/ReadVariableOpЂ(gaussian_noise_6/StatefulPartitionedCallЂ(gaussian_noise_7/StatefulPartitionedCallЂ(gaussian_noise_8/StatefulPartitionedCallЂ-layer_normalization_1/StatefulPartitionedCallй
-layer_normalization_1/StatefulPartitionedCallStatefulPartitionedCallinput_41layer_normalization_1_layer_normalization_1_gamma0layer_normalization_1_layer_normalization_1_beta*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџx'*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *X
fSRQ
O__inference_layer_normalization_1_layer_call_and_return_conditional_losses_8415Л
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_1/StatefulPartitionedCall:output:0conv2d_6_conv2d_6_kernelconv2d_6_conv2d_6_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_conv2d_6_layer_call_and_return_conditional_losses_8435є
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_8443
(gaussian_noise_6/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_gaussian_noise_6_layer_call_and_return_conditional_losses_8987і
activation_8/PartitionedCallPartitionedCall1gaussian_noise_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_activation_8_layer_call_and_return_conditional_losses_8456Њ
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall%activation_8/PartitionedCall:output:0conv2d_7_conv2d_7_kernelconv2d_7_conv2d_7_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ	*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_conv2d_7_layer_call_and_return_conditional_losses_8474с
flatten_2/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџа* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_flatten_2_layer_call_and_return_conditional_losses_8484б
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:03batch_normalization_batch_normalization_moving_mean7batch_normalization_batch_normalization_moving_variance-batch_normalization_batch_normalization_gamma,batch_normalization_batch_normalization_beta*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџа*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_8052я
reshape/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ	* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_reshape_layer_call_and_return_conditional_losses_8505Ј
(gaussian_noise_7/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0)^gaussian_noise_6/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ	* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_gaussian_noise_7_layer_call_and_return_conditional_losses_8883і
activation_9/PartitionedCallPartitionedCall1gaussian_noise_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ	* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_activation_9_layer_call_and_return_conditional_losses_8518Њ
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall%activation_9/PartitionedCall:output:0conv2d_8_conv2d_8_kernelconv2d_8_conv2d_8_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_conv2d_8_layer_call_and_return_conditional_losses_8536р
flatten_3/PartitionedCallPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџZ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_flatten_3_layer_call_and_return_conditional_losses_8546ф
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:07batch_normalization_1_batch_normalization_1_moving_mean;batch_normalization_1_batch_normalization_1_moving_variance1batch_normalization_1_batch_normalization_1_gamma0batch_normalization_1_batch_normalization_1_beta*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџZ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_8180ѕ
reshape_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_reshape_1_layer_call_and_return_conditional_losses_8567Њ
(gaussian_noise_8/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0)^gaussian_noise_7/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_gaussian_noise_8_layer_call_and_return_conditional_losses_8779ј
activation_10/PartitionedCallPartitionedCall1gaussian_noise_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_activation_10_layer_call_and_return_conditional_losses_8580н
flatten_4/PartitionedCallPartitionedCall&activation_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџZ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_flatten_4_layer_call_and_return_conditional_losses_8588ф
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:07batch_normalization_2_batch_normalization_2_moving_mean;batch_normalization_2_batch_normalization_2_moving_variance1batch_normalization_2_batch_normalization_2_gamma0batch_normalization_2_batch_normalization_2_beta*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџZ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_8308­
dense_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0dense_2_dense_2_kerneldense_2_dense_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ%*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_8611Ъ
+dense_2/ActivityRegularizer/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8 *6
f1R/
-__inference_dense_2_activity_regularizer_8626y
!dense_2/ActivityRegularizer/ShapeShape(dense_2/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:y
/dense_2/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1dense_2/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1dense_2/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:н
)dense_2/ActivityRegularizer/strided_sliceStridedSlice*dense_2/ActivityRegularizer/Shape:output:08dense_2/ActivityRegularizer/strided_slice/stack:output:0:dense_2/ActivityRegularizer/strided_slice/stack_1:output:0:dense_2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
 dense_2/ActivityRegularizer/CastCast2dense_2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: Ћ
#dense_2/ActivityRegularizer/truedivRealDiv4dense_2/ActivityRegularizer/PartitionedCall:output:0$dense_2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ч
activation_11/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ%* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_activation_11_layer_call_and_return_conditional_losses_8640
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_6_conv2d_6_kernel*&
_output_shapes
:		x*
dtype0
"conv2d_6/kernel/Regularizer/SquareSquare9conv2d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		xz
!conv2d_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
conv2d_6/kernel/Regularizer/SumSum&conv2d_6/kernel/Regularizer/Square:y:0*conv2d_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0(conv2d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
1conv2d_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_7_conv2d_7_kernel*&
_output_shapes
:*
dtype0
"conv2d_7/kernel/Regularizer/SquareSquare9conv2d_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
!conv2d_7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
conv2d_7/kernel/Regularizer/SumSum&conv2d_7/kernel/Regularizer/Square:y:0*conv2d_7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv2d_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:
conv2d_7/kernel/Regularizer/mulMul*conv2d_7/kernel/Regularizer/mul/x:output:0(conv2d_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
1conv2d_8/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_8_conv2d_8_kernel*&
_output_shapes
:*
dtype0
"conv2d_8/kernel/Regularizer/SquareSquare9conv2d_8/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
!conv2d_8/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
conv2d_8/kernel/Regularizer/SumSum&conv2d_8/kernel/Regularizer/Square:y:0*conv2d_8/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv2d_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:
conv2d_8/kernel/Regularizer/mulMul*conv2d_8/kernel/Regularizer/mul/x:output:0(conv2d_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_2_dense_2_kernel*
_output_shapes

:Z%*
dtype0
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:Z%q
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_2/kernel/Regularizer/SumSum%dense_2/kernel/Regularizer/Square:y:0)dense_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: u
IdentityIdentity&activation_11/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ%п
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall2^conv2d_6/kernel/Regularizer/Square/ReadVariableOp!^conv2d_7/StatefulPartitionedCall2^conv2d_7/kernel/Regularizer/Square/ReadVariableOp!^conv2d_8/StatefulPartitionedCall2^conv2d_8/kernel/Regularizer/Square/ReadVariableOp ^dense_2/StatefulPartitionedCall1^dense_2/kernel/Regularizer/Square/ReadVariableOp)^gaussian_noise_6/StatefulPartitionedCall)^gaussian_noise_7/StatefulPartitionedCall)^gaussian_noise_8/StatefulPartitionedCall.^layer_normalization_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*Z
_input_shapesI
G:џџџџџџџџџx': : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2f
1conv2d_6/kernel/Regularizer/Square/ReadVariableOp1conv2d_6/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2f
1conv2d_7/kernel/Regularizer/Square/ReadVariableOp1conv2d_7/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2f
1conv2d_8/kernel/Regularizer/Square/ReadVariableOp1conv2d_8/kernel/Regularizer/Square/ReadVariableOp2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2d
0dense_2/kernel/Regularizer/Square/ReadVariableOp0dense_2/kernel/Regularizer/Square/ReadVariableOp2T
(gaussian_noise_6/StatefulPartitionedCall(gaussian_noise_6/StatefulPartitionedCall2T
(gaussian_noise_7/StatefulPartitionedCall(gaussian_noise_7/StatefulPartitionedCall2T
(gaussian_noise_8/StatefulPartitionedCall(gaussian_noise_8/StatefulPartitionedCall2^
-layer_normalization_1/StatefulPartitionedCall-layer_normalization_1/StatefulPartitionedCall:X T
/
_output_shapes
:џџџџџџџџџx'
!
_user_specified_name	input_4
н

P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_10410

inputsL
>batchnorm_readvariableop_batch_normalization_1_moving_variance:ZF
8batchnorm_mul_readvariableop_batch_normalization_1_gamma:ZJ
<batchnorm_readvariableop_1_batch_normalization_1_moving_mean:ZC
5batchnorm_readvariableop_2_batch_normalization_1_beta:Z
identityЂbatchnorm/ReadVariableOpЂbatchnorm/ReadVariableOp_1Ђbatchnorm/ReadVariableOp_2Ђbatchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp>batchnorm_readvariableop_batch_normalization_1_moving_variance*
_output_shapes
:Z*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:ZP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:Z
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
:џџџџџџџџџZ
batchnorm/ReadVariableOp_1ReadVariableOp<batchnorm_readvariableop_1_batch_normalization_1_moving_mean*
_output_shapes
:Z*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:Z
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
:џџџџџџџџџZb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџZК
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџZ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџZ
 
_user_specified_nameinputs
Ѕ
L
0__inference_gaussian_noise_7_layer_call_fn_10302

inputs
identityТ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ	* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_gaussian_noise_7_layer_call_and_return_conditional_losses_8511h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ	"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ	:W S
/
_output_shapes
:џџџџџџџџџ	
 
_user_specified_nameinputs
р
Џ
A__inference_dense_2_layer_call_and_return_conditional_losses_8611

inputs6
$matmul_readvariableop_dense_2_kernel:Z%1
#biasadd_readvariableop_dense_2_bias:%
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂ0dense_2/kernel/Regularizer/Square/ReadVariableOpz
MatMul/ReadVariableOpReadVariableOp$matmul_readvariableop_dense_2_kernel*
_output_shapes

:Z%*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ%v
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_2_bias*
_output_shapes
:%*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ%
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp$matmul_readvariableop_dense_2_kernel*
_output_shapes

:Z%*
dtype0
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:Z%q
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_2/kernel/Regularizer/SumSum%dense_2/kernel/Regularizer/Square:y:0)dense_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ%Њ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp1^dense_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџZ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2d
0dense_2/kernel/Regularizer/Square/ReadVariableOp0dense_2/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџZ
 
_user_specified_nameinputs

Л
B__inference_conv2d_7_layer_call_and_return_conditional_losses_8474

inputs?
%conv2d_readvariableop_conv2d_7_kernel:2
$biasadd_readvariableop_conv2d_7_bias:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOpЂ1conv2d_7/kernel/Regularizer/Square/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOp%conv2d_readvariableop_conv2d_7_kernel*&
_output_shapes
:*
dtype0Б
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ	*
data_formatNCHW*
paddingVALID*
strides
w
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_conv2d_7_bias*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ	*
data_formatNCHW
1conv2d_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%conv2d_readvariableop_conv2d_7_kernel*&
_output_shapes
:*
dtype0
"conv2d_7/kernel/Regularizer/SquareSquare9conv2d_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
!conv2d_7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
conv2d_7/kernel/Regularizer/SumSum&conv2d_7/kernel/Regularizer/Square:y:0*conv2d_7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv2d_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:
conv2d_7/kernel/Regularizer/mulMul*conv2d_7/kernel/Regularizer/mul/x:output:0(conv2d_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ	Ћ
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp2^conv2d_7/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2f
1conv2d_7/kernel/Regularizer/Square/ReadVariableOp1conv2d_7/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_8137

inputsL
>batchnorm_readvariableop_batch_normalization_1_moving_variance:ZF
8batchnorm_mul_readvariableop_batch_normalization_1_gamma:ZJ
<batchnorm_readvariableop_1_batch_normalization_1_moving_mean:ZC
5batchnorm_readvariableop_2_batch_normalization_1_beta:Z
identityЂbatchnorm/ReadVariableOpЂbatchnorm/ReadVariableOp_1Ђbatchnorm/ReadVariableOp_2Ђbatchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp>batchnorm_readvariableop_batch_normalization_1_moving_variance*
_output_shapes
:Z*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:ZP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:Z
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
:џџџџџџџџџZ
batchnorm/ReadVariableOp_1ReadVariableOp<batchnorm_readvariableop_1_batch_normalization_1_moving_mean*
_output_shapes
:Z*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:Z
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
:џџџџџџџџџZb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџZК
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџZ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџZ
 
_user_specified_nameinputs


*__inference_CNN_2D_NORM_layer_call_fn_9391
input_41
layer_normalization_1_gamma:x'0
layer_normalization_1_beta:x')
conv2d_6_kernel:		x
conv2d_6_bias:)
conv2d_7_kernel:
conv2d_7_bias:.
batch_normalization_moving_mean:	а2
#batch_normalization_moving_variance:	а(
batch_normalization_gamma:	а'
batch_normalization_beta:	а)
conv2d_8_kernel:
conv2d_8_bias:/
!batch_normalization_1_moving_mean:Z3
%batch_normalization_1_moving_variance:Z)
batch_normalization_1_gamma:Z(
batch_normalization_1_beta:Z/
!batch_normalization_2_moving_mean:Z3
%batch_normalization_2_moving_variance:Z)
batch_normalization_2_gamma:Z(
batch_normalization_2_beta:Z 
dense_2_kernel:Z%
dense_2_bias:%
identityЂStatefulPartitionedCallІ
StatefulPartitionedCallStatefulPartitionedCallinput_4layer_normalization_1_gammalayer_normalization_1_betaconv2d_6_kernelconv2d_6_biasconv2d_7_kernelconv2d_7_biasbatch_normalization_moving_mean#batch_normalization_moving_variancebatch_normalization_gammabatch_normalization_betaconv2d_8_kernelconv2d_8_bias!batch_normalization_1_moving_mean%batch_normalization_1_moving_variancebatch_normalization_1_gammabatch_normalization_1_beta!batch_normalization_2_moving_mean%batch_normalization_2_moving_variancebatch_normalization_2_gammabatch_normalization_2_betadense_2_kerneldense_2_bias*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ%*2
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_CNN_2D_NORM_layer_call_and_return_conditional_losses_9181o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ%`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*Z
_input_shapesI
G:џџџџџџџџџx': : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:џџџџџџџџџx'
!
_user_specified_name	input_4
К
П
__inference_loss_fn_1_10635[
Aconv2d_7_kernel_regularizer_square_readvariableop_conv2d_7_kernel:
identityЂ1conv2d_7/kernel/Regularizer/Square/ReadVariableOpЛ
1conv2d_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOpAconv2d_7_kernel_regularizer_square_readvariableop_conv2d_7_kernel*&
_output_shapes
:*
dtype0
"conv2d_7/kernel/Regularizer/SquareSquare9conv2d_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
!conv2d_7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
conv2d_7/kernel/Regularizer/SumSum&conv2d_7/kernel/Regularizer/Square:y:0*conv2d_7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv2d_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:
conv2d_7/kernel/Regularizer/mulMul*conv2d_7/kernel/Regularizer/mul/x:output:0(conv2d_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentity#conv2d_7/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: z
NoOpNoOp2^conv2d_7/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*
_input_shapes
: 2f
1conv2d_7/kernel/Regularizer/Square/ReadVariableOp1conv2d_7/kernel/Regularizer/Square/ReadVariableOp
ъ
b
F__inference_activation_8_layer_call_and_return_conditional_losses_8456

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:џџџџџџџџџb
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
І	
i
J__inference_gaussian_noise_8_layer_call_and_return_conditional_losses_8779

inputs
identity;
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
 *ЭЬЬ=
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
dtype0
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*/
_output_shapes
:џџџџџџџџџa
addAddV2inputsrandom_normal:z:0*
T0*/
_output_shapes
:џџџџџџџџџW
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
к
Љ
(__inference_conv2d_8_layer_call_fn_10345

inputs)
conv2d_8_kernel:
conv2d_8_bias:
identityЂStatefulPartitionedCall№
StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_8_kernelconv2d_8_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_conv2d_8_layer_call_and_return_conditional_losses_8536w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ	: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ	
 
_user_specified_nameinputs
Ѕ
L
0__inference_gaussian_noise_8_layer_call_fn_10468

inputs
identityТ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_gaussian_noise_8_layer_call_and_return_conditional_losses_8573h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
І	
Ѓ
3__inference_batch_normalization_layer_call_fn_10215

inputs2
#batch_normalization_moving_variance:	а(
batch_normalization_gamma:	а.
batch_normalization_moving_mean:	а'
batch_normalization_beta:	а
identityЂStatefulPartitionedCallг
StatefulPartitionedCallStatefulPartitionedCallinputs#batch_normalization_moving_variancebatch_normalization_gammabatch_normalization_moving_meanbatch_normalization_beta*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџа*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_8009p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџа`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџа: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџа
 
_user_specified_nameinputs

E
)__inference_flatten_4_layer_call_fn_10503

inputs
identityГ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџZ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_flatten_4_layer_call_and_return_conditional_losses_8588`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџZ"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ё
`
D__inference_reshape_1_layer_call_and_return_conditional_losses_10463

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
valueB:б
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
value	B :Љ
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ`
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџZ:O K
'
_output_shapes
:џџџџџџџџџZ
 
_user_specified_nameinputs
єп
Л
E__inference_CNN_2D_NORM_layer_call_and_return_conditional_losses_9821

inputs^
Hlayer_normalization_1_reshape_readvariableop_layer_normalization_1_gamma:x'_
Ilayer_normalization_1_reshape_1_readvariableop_layer_normalization_1_beta:x'H
.conv2d_6_conv2d_readvariableop_conv2d_6_kernel:		x;
-conv2d_6_biasadd_readvariableop_conv2d_6_bias:H
.conv2d_7_conv2d_readvariableop_conv2d_7_kernel:;
-conv2d_7_biasadd_readvariableop_conv2d_7_bias:_
Pbatch_normalization_batchnorm_readvariableop_batch_normalization_moving_variance:	аY
Jbatch_normalization_batchnorm_mul_readvariableop_batch_normalization_gamma:	а]
Nbatch_normalization_batchnorm_readvariableop_1_batch_normalization_moving_mean:	аV
Gbatch_normalization_batchnorm_readvariableop_2_batch_normalization_beta:	аH
.conv2d_8_conv2d_readvariableop_conv2d_8_kernel:;
-conv2d_8_biasadd_readvariableop_conv2d_8_bias:b
Tbatch_normalization_1_batchnorm_readvariableop_batch_normalization_1_moving_variance:Z\
Nbatch_normalization_1_batchnorm_mul_readvariableop_batch_normalization_1_gamma:Z`
Rbatch_normalization_1_batchnorm_readvariableop_1_batch_normalization_1_moving_mean:ZY
Kbatch_normalization_1_batchnorm_readvariableop_2_batch_normalization_1_beta:Zb
Tbatch_normalization_2_batchnorm_readvariableop_batch_normalization_2_moving_variance:Z\
Nbatch_normalization_2_batchnorm_mul_readvariableop_batch_normalization_2_gamma:Z`
Rbatch_normalization_2_batchnorm_readvariableop_1_batch_normalization_2_moving_mean:ZY
Kbatch_normalization_2_batchnorm_readvariableop_2_batch_normalization_2_beta:Z>
,dense_2_matmul_readvariableop_dense_2_kernel:Z%9
+dense_2_biasadd_readvariableop_dense_2_bias:%
identityЂ,batch_normalization/batchnorm/ReadVariableOpЂ.batch_normalization/batchnorm/ReadVariableOp_1Ђ.batch_normalization/batchnorm/ReadVariableOp_2Ђ0batch_normalization/batchnorm/mul/ReadVariableOpЂ.batch_normalization_1/batchnorm/ReadVariableOpЂ0batch_normalization_1/batchnorm/ReadVariableOp_1Ђ0batch_normalization_1/batchnorm/ReadVariableOp_2Ђ2batch_normalization_1/batchnorm/mul/ReadVariableOpЂ.batch_normalization_2/batchnorm/ReadVariableOpЂ0batch_normalization_2/batchnorm/ReadVariableOp_1Ђ0batch_normalization_2/batchnorm/ReadVariableOp_2Ђ2batch_normalization_2/batchnorm/mul/ReadVariableOpЂconv2d_6/BiasAdd/ReadVariableOpЂconv2d_6/Conv2D/ReadVariableOpЂ1conv2d_6/kernel/Regularizer/Square/ReadVariableOpЂconv2d_7/BiasAdd/ReadVariableOpЂconv2d_7/Conv2D/ReadVariableOpЂ1conv2d_7/kernel/Regularizer/Square/ReadVariableOpЂconv2d_8/BiasAdd/ReadVariableOpЂconv2d_8/Conv2D/ReadVariableOpЂ1conv2d_8/kernel/Regularizer/Square/ReadVariableOpЂdense_2/BiasAdd/ReadVariableOpЂdense_2/MatMul/ReadVariableOpЂ0dense_2/kernel/Regularizer/Square/ReadVariableOpЂ,layer_normalization_1/Reshape/ReadVariableOpЂ.layer_normalization_1/Reshape_1/ReadVariableOp
4layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         М
"layer_normalization_1/moments/meanMeaninputs=layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
	keep_dims(Ё
*layer_normalization_1/moments/StopGradientStopGradient+layer_normalization_1/moments/mean:output:0*
T0*/
_output_shapes
:џџџџџџџџџЛ
/layer_normalization_1/moments/SquaredDifferenceSquaredDifferenceinputs3layer_normalization_1/moments/StopGradient:output:0*
T0*/
_output_shapes
:џџџџџџџџџx'
8layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         ё
&layer_normalization_1/moments/varianceMean3layer_normalization_1/moments/SquaredDifference:z:0Alayer_normalization_1/moments/variance/reduction_indices:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
	keep_dims(Й
,layer_normalization_1/Reshape/ReadVariableOpReadVariableOpHlayer_normalization_1_reshape_readvariableop_layer_normalization_1_gamma*"
_output_shapes
:x'*
dtype0|
#layer_normalization_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   x      '   Н
layer_normalization_1/ReshapeReshape4layer_normalization_1/Reshape/ReadVariableOp:value:0,layer_normalization_1/Reshape/shape:output:0*
T0*&
_output_shapes
:x'М
.layer_normalization_1/Reshape_1/ReadVariableOpReadVariableOpIlayer_normalization_1_reshape_1_readvariableop_layer_normalization_1_beta*"
_output_shapes
:x'*
dtype0~
%layer_normalization_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   x      '   У
layer_normalization_1/Reshape_1Reshape6layer_normalization_1/Reshape_1/ReadVariableOp:value:0.layer_normalization_1/Reshape_1/shape:output:0*
T0*&
_output_shapes
:x'j
%layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Пж3Ч
#layer_normalization_1/batchnorm/addAddV2/layer_normalization_1/moments/variance:output:0.layer_normalization_1/batchnorm/add/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
%layer_normalization_1/batchnorm/RsqrtRsqrt'layer_normalization_1/batchnorm/add:z:0*
T0*/
_output_shapes
:џџџџџџџџџЗ
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0&layer_normalization_1/Reshape:output:0*
T0*/
_output_shapes
:џџџџџџџџџx'
%layer_normalization_1/batchnorm/mul_1Mulinputs'layer_normalization_1/batchnorm/mul:z:0*
T0*/
_output_shapes
:џџџџџџџџџx'М
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*/
_output_shapes
:џџџџџџџџџx'Й
#layer_normalization_1/batchnorm/subSub(layer_normalization_1/Reshape_1:output:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*/
_output_shapes
:џџџџџџџџџx'М
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*/
_output_shapes
:џџџџџџџџџx'
conv2d_6/Conv2D/ReadVariableOpReadVariableOp.conv2d_6_conv2d_readvariableop_conv2d_6_kernel*&
_output_shapes
:		x*
dtype0ц
conv2d_6/Conv2DConv2D)layer_normalization_1/batchnorm/add_1:z:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
data_formatNCHW*
paddingVALID*
strides

conv2d_6/BiasAdd/ReadVariableOpReadVariableOp-conv2d_6_biasadd_readvariableop_conv2d_6_bias*
_output_shapes
:*
dtype0Џ
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
data_formatNCHWС
max_pooling2d_2/MaxPoolMaxPoolconv2d_6/BiasAdd:output:0*/
_output_shapes
:џџџџџџџџџ*
data_formatNCHW*
ksize
*
paddingVALID*
strides
u
activation_8/ReluRelu max_pooling2d_2/MaxPool:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
conv2d_7/Conv2D/ReadVariableOpReadVariableOp.conv2d_7_conv2d_readvariableop_conv2d_7_kernel*&
_output_shapes
:*
dtype0м
conv2d_7/Conv2DConv2Dactivation_8/Relu:activations:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ	*
data_formatNCHW*
paddingVALID*
strides

conv2d_7/BiasAdd/ReadVariableOpReadVariableOp-conv2d_7_biasadd_readvariableop_conv2d_7_bias*
_output_shapes
:*
dtype0Џ
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ	*
data_formatNCHW`
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџа  
flatten_2/ReshapeReshapeconv2d_7/BiasAdd:output:0flatten_2/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџаК
,batch_normalization/batchnorm/ReadVariableOpReadVariableOpPbatch_normalization_batchnorm_readvariableop_batch_normalization_moving_variance*
_output_shapes	
:а*
dtype0h
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:Д
!batch_normalization/batchnorm/addAddV24batch_normalization/batchnorm/ReadVariableOp:value:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:аy
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:аИ
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOpJbatch_normalization_batchnorm_mul_readvariableop_batch_normalization_gamma*
_output_shapes	
:а*
dtype0Б
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:а 
#batch_normalization/batchnorm/mul_1Mulflatten_2/Reshape:output:0%batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:џџџџџџџџџаК
.batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOpNbatch_normalization_batchnorm_readvariableop_1_batch_normalization_moving_mean*
_output_shapes	
:а*
dtype0Џ
#batch_normalization/batchnorm/mul_2Mul6batch_normalization/batchnorm/ReadVariableOp_1:value:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:аГ
.batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOpGbatch_normalization_batchnorm_readvariableop_2_batch_normalization_beta*
_output_shapes	
:а*
dtype0Џ
!batch_normalization/batchnorm/subSub6batch_normalization/batchnorm/ReadVariableOp_2:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:аЏ
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:џџџџџџџџџаd
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
valueB:љ
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
value	B :	б
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:
reshape/ReshapeReshape'batch_normalization/batchnorm/add_1:z:0reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ	m
activation_9/ReluRelureshape/Reshape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ	
conv2d_8/Conv2D/ReadVariableOpReadVariableOp.conv2d_8_conv2d_readvariableop_conv2d_8_kernel*&
_output_shapes
:*
dtype0м
conv2d_8/Conv2DConv2Dactivation_9/Relu:activations:0&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
data_formatNCHW*
paddingVALID*
strides

conv2d_8/BiasAdd/ReadVariableOpReadVariableOp-conv2d_8_biasadd_readvariableop_conv2d_8_bias*
_output_shapes
:*
dtype0Џ
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
data_formatNCHW`
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџZ   
flatten_3/ReshapeReshapeconv2d_8/BiasAdd:output:0flatten_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџZП
.batch_normalization_1/batchnorm/ReadVariableOpReadVariableOpTbatch_normalization_1_batchnorm_readvariableop_batch_normalization_1_moving_variance*
_output_shapes
:Z*
dtype0j
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:Й
#batch_normalization_1/batchnorm/addAddV26batch_normalization_1/batchnorm/ReadVariableOp:value:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes
:Z|
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
:ZН
2batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpNbatch_normalization_1_batchnorm_mul_readvariableop_batch_normalization_1_gamma*
_output_shapes
:Z*
dtype0Ж
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:0:batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:ZЃ
%batch_normalization_1/batchnorm/mul_1Mulflatten_3/Reshape:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџZП
0batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOpRbatch_normalization_1_batchnorm_readvariableop_1_batch_normalization_1_moving_mean*
_output_shapes
:Z*
dtype0Д
%batch_normalization_1/batchnorm/mul_2Mul8batch_normalization_1/batchnorm/ReadVariableOp_1:value:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
:ZИ
0batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOpKbatch_normalization_1_batchnorm_readvariableop_2_batch_normalization_1_beta*
_output_shapes
:Z*
dtype0Д
#batch_normalization_1/batchnorm/subSub8batch_normalization_1/batchnorm/ReadVariableOp_2:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:ZД
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџZh
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
valueB:
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
value	B :л
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0"reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:Ѓ
reshape_1/ReshapeReshape)batch_normalization_1/batchnorm/add_1:z:0 reshape_1/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџp
activation_10/ReluRelureshape_1/Reshape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ`
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџZ   
flatten_4/ReshapeReshape activation_10/Relu:activations:0flatten_4/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџZП
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOpTbatch_normalization_2_batchnorm_readvariableop_batch_normalization_2_moving_variance*
_output_shapes
:Z*
dtype0j
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:Й
#batch_normalization_2/batchnorm/addAddV26batch_normalization_2/batchnorm/ReadVariableOp:value:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:Z|
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:ZН
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpNbatch_normalization_2_batchnorm_mul_readvariableop_batch_normalization_2_gamma*
_output_shapes
:Z*
dtype0Ж
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:ZЃ
%batch_normalization_2/batchnorm/mul_1Mulflatten_4/Reshape:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџZП
0batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOpRbatch_normalization_2_batchnorm_readvariableop_1_batch_normalization_2_moving_mean*
_output_shapes
:Z*
dtype0Д
%batch_normalization_2/batchnorm/mul_2Mul8batch_normalization_2/batchnorm/ReadVariableOp_1:value:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:ZИ
0batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOpKbatch_normalization_2_batchnorm_readvariableop_2_batch_normalization_2_beta*
_output_shapes
:Z*
dtype0Д
#batch_normalization_2/batchnorm/subSub8batch_normalization_2/batchnorm/ReadVariableOp_2:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:ZД
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџZ
dense_2/MatMul/ReadVariableOpReadVariableOp,dense_2_matmul_readvariableop_dense_2_kernel*
_output_shapes

:Z%*
dtype0
dense_2/MatMulMatMul)batch_normalization_2/batchnorm/add_1:z:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ%
dense_2/BiasAdd/ReadVariableOpReadVariableOp+dense_2_biasadd_readvariableop_dense_2_bias*
_output_shapes
:%*
dtype0
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ%r
dense_2/ActivityRegularizer/AbsAbsdense_2/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ%r
!dense_2/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_2/ActivityRegularizer/SumSum#dense_2/ActivityRegularizer/Abs:y:0*dense_2/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_2/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:
dense_2/ActivityRegularizer/mulMul*dense_2/ActivityRegularizer/mul/x:output:0(dense_2/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: i
!dense_2/ActivityRegularizer/ShapeShapedense_2/BiasAdd:output:0*
T0*
_output_shapes
:y
/dense_2/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1dense_2/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1dense_2/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:н
)dense_2/ActivityRegularizer/strided_sliceStridedSlice*dense_2/ActivityRegularizer/Shape:output:08dense_2/ActivityRegularizer/strided_slice/stack:output:0:dense_2/ActivityRegularizer/strided_slice/stack_1:output:0:dense_2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
 dense_2/ActivityRegularizer/CastCast2dense_2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 
#dense_2/ActivityRegularizer/truedivRealDiv#dense_2/ActivityRegularizer/mul:z:0$dense_2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: n
activation_11/SoftplusSoftplusdense_2/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ%Ј
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOp.conv2d_6_conv2d_readvariableop_conv2d_6_kernel*&
_output_shapes
:		x*
dtype0
"conv2d_6/kernel/Regularizer/SquareSquare9conv2d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		xz
!conv2d_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
conv2d_6/kernel/Regularizer/SumSum&conv2d_6/kernel/Regularizer/Square:y:0*conv2d_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0(conv2d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: Ј
1conv2d_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOp.conv2d_7_conv2d_readvariableop_conv2d_7_kernel*&
_output_shapes
:*
dtype0
"conv2d_7/kernel/Regularizer/SquareSquare9conv2d_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
!conv2d_7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
conv2d_7/kernel/Regularizer/SumSum&conv2d_7/kernel/Regularizer/Square:y:0*conv2d_7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv2d_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:
conv2d_7/kernel/Regularizer/mulMul*conv2d_7/kernel/Regularizer/mul/x:output:0(conv2d_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: Ј
1conv2d_8/kernel/Regularizer/Square/ReadVariableOpReadVariableOp.conv2d_8_conv2d_readvariableop_conv2d_8_kernel*&
_output_shapes
:*
dtype0
"conv2d_8/kernel/Regularizer/SquareSquare9conv2d_8/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
!conv2d_8/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
conv2d_8/kernel/Regularizer/SumSum&conv2d_8/kernel/Regularizer/Square:y:0*conv2d_8/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv2d_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:
conv2d_8/kernel/Regularizer/mulMul*conv2d_8/kernel/Regularizer/mul/x:output:0(conv2d_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp,dense_2_matmul_readvariableop_dense_2_kernel*
_output_shapes

:Z%*
dtype0
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:Z%q
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_2/kernel/Regularizer/SumSum%dense_2/kernel/Regularizer/Square:y:0)dense_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: s
IdentityIdentity$activation_11/Softplus:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ%л	
NoOpNoOp-^batch_normalization/batchnorm/ReadVariableOp/^batch_normalization/batchnorm/ReadVariableOp_1/^batch_normalization/batchnorm/ReadVariableOp_21^batch_normalization/batchnorm/mul/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp1^batch_normalization_1/batchnorm/ReadVariableOp_11^batch_normalization_1/batchnorm/ReadVariableOp_23^batch_normalization_1/batchnorm/mul/ReadVariableOp/^batch_normalization_2/batchnorm/ReadVariableOp1^batch_normalization_2/batchnorm/ReadVariableOp_11^batch_normalization_2/batchnorm/ReadVariableOp_23^batch_normalization_2/batchnorm/mul/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp2^conv2d_6/kernel/Regularizer/Square/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp2^conv2d_7/kernel/Regularizer/Square/ReadVariableOp ^conv2d_8/BiasAdd/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp2^conv2d_8/kernel/Regularizer/Square/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp1^dense_2/kernel/Regularizer/Square/ReadVariableOp-^layer_normalization_1/Reshape/ReadVariableOp/^layer_normalization_1/Reshape_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*Z
_input_shapesI
G:џџџџџџџџџx': : : : : : : : : : : : : : : : : : : : : : 2\
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
2batch_normalization_2/batchnorm/mul/ReadVariableOp2batch_normalization_2/batchnorm/mul/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2f
1conv2d_6/kernel/Regularizer/Square/ReadVariableOp1conv2d_6/kernel/Regularizer/Square/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp2f
1conv2d_7/kernel/Regularizer/Square/ReadVariableOp1conv2d_7/kernel/Regularizer/Square/ReadVariableOp2B
conv2d_8/BiasAdd/ReadVariableOpconv2d_8/BiasAdd/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp2f
1conv2d_8/kernel/Regularizer/Square/ReadVariableOp1conv2d_8/kernel/Regularizer/Square/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2d
0dense_2/kernel/Regularizer/Square/ReadVariableOp0dense_2/kernel/Regularizer/Square/ReadVariableOp2\
,layer_normalization_1/Reshape/ReadVariableOp,layer_normalization_1/Reshape/ReadVariableOp2`
.layer_normalization_1/Reshape_1/ReadVariableOp.layer_normalization_1/Reshape_1/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџx'
 
_user_specified_nameinputs
ѕ
М
C__inference_conv2d_7_layer_call_and_return_conditional_losses_10195

inputs?
%conv2d_readvariableop_conv2d_7_kernel:2
$biasadd_readvariableop_conv2d_7_bias:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOpЂ1conv2d_7/kernel/Regularizer/Square/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOp%conv2d_readvariableop_conv2d_7_kernel*&
_output_shapes
:*
dtype0Б
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ	*
data_formatNCHW*
paddingVALID*
strides
w
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_conv2d_7_bias*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ	*
data_formatNCHW
1conv2d_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%conv2d_readvariableop_conv2d_7_kernel*&
_output_shapes
:*
dtype0
"conv2d_7/kernel/Regularizer/SquareSquare9conv2d_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
!conv2d_7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
conv2d_7/kernel/Regularizer/SumSum&conv2d_7/kernel/Regularizer/Square:y:0*conv2d_7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv2d_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:
conv2d_7/kernel/Regularizer/mulMul*conv2d_7/kernel/Regularizer/mul/x:output:0(conv2d_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ	Ћ
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp2^conv2d_7/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2f
1conv2d_7/kernel/Regularizer/Square/ReadVariableOp1conv2d_7/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ъ
]
A__inference_reshape_layer_call_and_return_conditional_losses_8505

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
valueB:б
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
value	B :	Љ
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ	`
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџа:P L
(
_output_shapes
:џџџџџџџџџа
 
_user_specified_nameinputs
н

N__inference_batch_normalization_layer_call_and_return_conditional_losses_10244

inputsK
<batchnorm_readvariableop_batch_normalization_moving_variance:	аE
6batchnorm_mul_readvariableop_batch_normalization_gamma:	аI
:batchnorm_readvariableop_1_batch_normalization_moving_mean:	аB
3batchnorm_readvariableop_2_batch_normalization_beta:	а
identityЂbatchnorm/ReadVariableOpЂbatchnorm/ReadVariableOp_1Ђbatchnorm/ReadVariableOp_2Ђbatchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp<batchnorm_readvariableop_batch_normalization_moving_variance*
_output_shapes	
:а*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:аQ
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:а
batchnorm/mul/ReadVariableOpReadVariableOp6batchnorm_mul_readvariableop_batch_normalization_gamma*
_output_shapes	
:а*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:аd
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:џџџџџџџџџа
batchnorm/ReadVariableOp_1ReadVariableOp:batchnorm_readvariableop_1_batch_normalization_moving_mean*
_output_shapes	
:а*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:а
batchnorm/ReadVariableOp_2ReadVariableOp3batchnorm_readvariableop_2_batch_normalization_beta*
_output_shapes	
:а*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:аs
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:џџџџџџџџџаc
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџаК
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџа: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџа
 
_user_specified_nameinputs
ї
i
0__inference_gaussian_noise_7_layer_call_fn_10307

inputs
identityЂStatefulPartitionedCallв
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ	* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_gaussian_noise_7_layer_call_and_return_conditional_losses_8883w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ	22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ	
 
_user_specified_nameinputs

`
D__inference_flatten_2_layer_call_and_return_conditional_losses_10206

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџа  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџаY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџа"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ	:W S
/
_output_shapes
:џџџџџџџџџ	
 
_user_specified_nameinputs
М
e
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_8443

inputs
identity
MaxPoolMaxPoolinputs*/
_output_shapes
:џџџџџџџџџ*
data_formatNCHW*
ksize
*
paddingVALID*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

H
,__inference_activation_8_layer_call_fn_10161

inputs
identityО
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_activation_8_layer_call_and_return_conditional_losses_8456h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


*__inference_CNN_2D_NORM_layer_call_fn_8692
input_41
layer_normalization_1_gamma:x'0
layer_normalization_1_beta:x')
conv2d_6_kernel:		x
conv2d_6_bias:)
conv2d_7_kernel:
conv2d_7_bias:2
#batch_normalization_moving_variance:	а(
batch_normalization_gamma:	а.
batch_normalization_moving_mean:	а'
batch_normalization_beta:	а)
conv2d_8_kernel:
conv2d_8_bias:3
%batch_normalization_1_moving_variance:Z)
batch_normalization_1_gamma:Z/
!batch_normalization_1_moving_mean:Z(
batch_normalization_1_beta:Z3
%batch_normalization_2_moving_variance:Z)
batch_normalization_2_gamma:Z/
!batch_normalization_2_moving_mean:Z(
batch_normalization_2_beta:Z 
dense_2_kernel:Z%
dense_2_bias:%
identityЂStatefulPartitionedCallЌ
StatefulPartitionedCallStatefulPartitionedCallinput_4layer_normalization_1_gammalayer_normalization_1_betaconv2d_6_kernelconv2d_6_biasconv2d_7_kernelconv2d_7_bias#batch_normalization_moving_variancebatch_normalization_gammabatch_normalization_moving_meanbatch_normalization_betaconv2d_8_kernelconv2d_8_bias%batch_normalization_1_moving_variancebatch_normalization_1_gamma!batch_normalization_1_moving_meanbatch_normalization_1_beta%batch_normalization_2_moving_variancebatch_normalization_2_gamma!batch_normalization_2_moving_meanbatch_normalization_2_betadense_2_kerneldense_2_bias*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ%*8
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_CNN_2D_NORM_layer_call_and_return_conditional_losses_8667o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ%`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*Z
_input_shapesI
G:џџџџџџџџџx': : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:џџџџџџџџџx'
!
_user_specified_name	input_4
к
Љ
(__inference_conv2d_7_layer_call_fn_10179

inputs)
conv2d_7_kernel:
conv2d_7_bias:
identityЂStatefulPartitionedCall№
StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_7_kernelconv2d_7_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ	*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_conv2d_7_layer_call_and_return_conditional_losses_8474w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

H
,__inference_activation_9_layer_call_fn_10327

inputs
identityО
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ	* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_activation_9_layer_call_and_return_conditional_losses_8518h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ	"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ	:W S
/
_output_shapes
:џџџџџџџџџ	
 
_user_specified_nameinputs
щ
g
K__inference_gaussian_noise_8_layer_call_and_return_conditional_losses_10477

inputs
identityV
IdentityIdentityinputs*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

`
D__inference_flatten_4_layer_call_and_return_conditional_losses_10509

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџZ   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:џџџџџџџџџZX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџZ"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
е&
Ф
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_10581

inputsN
@assignmovingavg_readvariableop_batch_normalization_2_moving_mean:ZT
Fassignmovingavg_1_readvariableop_batch_normalization_2_moving_variance:ZF
8batchnorm_mul_readvariableop_batch_normalization_2_gamma:ZA
3batchnorm_readvariableop_batch_normalization_2_beta:Z
identityЂAssignMovingAvgЂAssignMovingAvg/ReadVariableOpЂAssignMovingAvg_1Ђ AssignMovingAvg_1/ReadVariableOpЂbatchnorm/ReadVariableOpЂbatchnorm/mul/ReadVariableOph
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

:Z
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:џџџџџџџџџZl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
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
з#<
AssignMovingAvg/ReadVariableOpReadVariableOp@assignmovingavg_readvariableop_batch_normalization_2_moving_mean*
_output_shapes
:Z*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:Zx
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:ZХ
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
з#<Ѓ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpFassignmovingavg_1_readvariableop_batch_normalization_2_moving_variance*
_output_shapes
:Z*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:Z~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:Zб
AssignMovingAvg_1AssignSubVariableOpFassignmovingavg_1_readvariableop_batch_normalization_2_moving_varianceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:ZP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:Z
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
:џџџџџџџџџZh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:Z
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
:џџџџџџџџџZb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџZъ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџZ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџZ
 
_user_specified_nameinputs
Є	
Ѓ
3__inference_batch_normalization_layer_call_fn_10224

inputs.
batch_normalization_moving_mean:	а2
#batch_normalization_moving_variance:	а(
batch_normalization_gamma:	а'
batch_normalization_beta:	а
identityЂStatefulPartitionedCallб
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_moving_mean#batch_normalization_moving_variancebatch_normalization_gammabatch_normalization_beta*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџа*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_8052p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџа`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџа: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџа
 
_user_specified_nameinputs

I
-__inference_activation_10_layer_call_fn_10493

inputs
identityП
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_activation_10_layer_call_and_return_conditional_losses_8580h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
џ
I
-__inference_activation_11_layer_call_fn_10608

inputs
identityЗ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ%* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_activation_11_layer_call_and_return_conditional_losses_8640`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ%"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ%:O K
'
_output_shapes
:џџџџџџџџџ%
 
_user_specified_nameinputs


E__inference_CNN_2D_NORM_layer_call_and_return_conditional_losses_8667

inputsG
1layer_normalization_1_layer_normalization_1_gamma:x'F
0layer_normalization_1_layer_normalization_1_beta:x'2
conv2d_6_conv2d_6_kernel:		x$
conv2d_6_conv2d_6_bias:2
conv2d_7_conv2d_7_kernel:$
conv2d_7_conv2d_7_bias:F
7batch_normalization_batch_normalization_moving_variance:	а<
-batch_normalization_batch_normalization_gamma:	аB
3batch_normalization_batch_normalization_moving_mean:	а;
,batch_normalization_batch_normalization_beta:	а2
conv2d_8_conv2d_8_kernel:$
conv2d_8_conv2d_8_bias:I
;batch_normalization_1_batch_normalization_1_moving_variance:Z?
1batch_normalization_1_batch_normalization_1_gamma:ZE
7batch_normalization_1_batch_normalization_1_moving_mean:Z>
0batch_normalization_1_batch_normalization_1_beta:ZI
;batch_normalization_2_batch_normalization_2_moving_variance:Z?
1batch_normalization_2_batch_normalization_2_gamma:ZE
7batch_normalization_2_batch_normalization_2_moving_mean:Z>
0batch_normalization_2_batch_normalization_2_beta:Z(
dense_2_dense_2_kernel:Z%"
dense_2_dense_2_bias:%
identityЂ+batch_normalization/StatefulPartitionedCallЂ-batch_normalization_1/StatefulPartitionedCallЂ-batch_normalization_2/StatefulPartitionedCallЂ conv2d_6/StatefulPartitionedCallЂ1conv2d_6/kernel/Regularizer/Square/ReadVariableOpЂ conv2d_7/StatefulPartitionedCallЂ1conv2d_7/kernel/Regularizer/Square/ReadVariableOpЂ conv2d_8/StatefulPartitionedCallЂ1conv2d_8/kernel/Regularizer/Square/ReadVariableOpЂdense_2/StatefulPartitionedCallЂ0dense_2/kernel/Regularizer/Square/ReadVariableOpЂ-layer_normalization_1/StatefulPartitionedCallи
-layer_normalization_1/StatefulPartitionedCallStatefulPartitionedCallinputs1layer_normalization_1_layer_normalization_1_gamma0layer_normalization_1_layer_normalization_1_beta*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџx'*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *X
fSRQ
O__inference_layer_normalization_1_layer_call_and_return_conditional_losses_8415Л
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_1/StatefulPartitionedCall:output:0conv2d_6_conv2d_6_kernelconv2d_6_conv2d_6_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_conv2d_6_layer_call_and_return_conditional_losses_8435є
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_8443ѕ
 gaussian_noise_6/PartitionedCallPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_gaussian_noise_6_layer_call_and_return_conditional_losses_8449ю
activation_8/PartitionedCallPartitionedCall)gaussian_noise_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_activation_8_layer_call_and_return_conditional_losses_8456Њ
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall%activation_8/PartitionedCall:output:0conv2d_7_conv2d_7_kernelconv2d_7_conv2d_7_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ	*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_conv2d_7_layer_call_and_return_conditional_losses_8474с
flatten_2/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџа* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_flatten_2_layer_call_and_return_conditional_losses_8484г
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:07batch_normalization_batch_normalization_moving_variance-batch_normalization_batch_normalization_gamma3batch_normalization_batch_normalization_moving_mean,batch_normalization_batch_normalization_beta*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџа*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_8009я
reshape/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ	* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_reshape_layer_call_and_return_conditional_losses_8505э
 gaussian_noise_7/PartitionedCallPartitionedCall reshape/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ	* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_gaussian_noise_7_layer_call_and_return_conditional_losses_8511ю
activation_9/PartitionedCallPartitionedCall)gaussian_noise_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ	* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_activation_9_layer_call_and_return_conditional_losses_8518Њ
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall%activation_9/PartitionedCall:output:0conv2d_8_conv2d_8_kernelconv2d_8_conv2d_8_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_conv2d_8_layer_call_and_return_conditional_losses_8536р
flatten_3/PartitionedCallPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџZ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_flatten_3_layer_call_and_return_conditional_losses_8546ц
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0;batch_normalization_1_batch_normalization_1_moving_variance1batch_normalization_1_batch_normalization_1_gamma7batch_normalization_1_batch_normalization_1_moving_mean0batch_normalization_1_batch_normalization_1_beta*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџZ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_8137ѕ
reshape_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_reshape_1_layer_call_and_return_conditional_losses_8567я
 gaussian_noise_8/PartitionedCallPartitionedCall"reshape_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_gaussian_noise_8_layer_call_and_return_conditional_losses_8573№
activation_10/PartitionedCallPartitionedCall)gaussian_noise_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_activation_10_layer_call_and_return_conditional_losses_8580н
flatten_4/PartitionedCallPartitionedCall&activation_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџZ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_flatten_4_layer_call_and_return_conditional_losses_8588ц
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0;batch_normalization_2_batch_normalization_2_moving_variance1batch_normalization_2_batch_normalization_2_gamma7batch_normalization_2_batch_normalization_2_moving_mean0batch_normalization_2_batch_normalization_2_beta*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџZ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_8265­
dense_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0dense_2_dense_2_kerneldense_2_dense_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ%*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_8611Ъ
+dense_2/ActivityRegularizer/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8 *6
f1R/
-__inference_dense_2_activity_regularizer_8626y
!dense_2/ActivityRegularizer/ShapeShape(dense_2/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:y
/dense_2/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1dense_2/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1dense_2/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:н
)dense_2/ActivityRegularizer/strided_sliceStridedSlice*dense_2/ActivityRegularizer/Shape:output:08dense_2/ActivityRegularizer/strided_slice/stack:output:0:dense_2/ActivityRegularizer/strided_slice/stack_1:output:0:dense_2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
 dense_2/ActivityRegularizer/CastCast2dense_2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: Ћ
#dense_2/ActivityRegularizer/truedivRealDiv4dense_2/ActivityRegularizer/PartitionedCall:output:0$dense_2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ч
activation_11/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ%* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_activation_11_layer_call_and_return_conditional_losses_8640
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_6_conv2d_6_kernel*&
_output_shapes
:		x*
dtype0
"conv2d_6/kernel/Regularizer/SquareSquare9conv2d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		xz
!conv2d_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
conv2d_6/kernel/Regularizer/SumSum&conv2d_6/kernel/Regularizer/Square:y:0*conv2d_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0(conv2d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
1conv2d_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_7_conv2d_7_kernel*&
_output_shapes
:*
dtype0
"conv2d_7/kernel/Regularizer/SquareSquare9conv2d_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
!conv2d_7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
conv2d_7/kernel/Regularizer/SumSum&conv2d_7/kernel/Regularizer/Square:y:0*conv2d_7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv2d_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:
conv2d_7/kernel/Regularizer/mulMul*conv2d_7/kernel/Regularizer/mul/x:output:0(conv2d_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
1conv2d_8/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_8_conv2d_8_kernel*&
_output_shapes
:*
dtype0
"conv2d_8/kernel/Regularizer/SquareSquare9conv2d_8/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
!conv2d_8/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
conv2d_8/kernel/Regularizer/SumSum&conv2d_8/kernel/Regularizer/Square:y:0*conv2d_8/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv2d_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:
conv2d_8/kernel/Regularizer/mulMul*conv2d_8/kernel/Regularizer/mul/x:output:0(conv2d_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_2_dense_2_kernel*
_output_shapes

:Z%*
dtype0
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:Z%q
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_2/kernel/Regularizer/SumSum%dense_2/kernel/Regularizer/Square:y:0)dense_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: u
IdentityIdentity&activation_11/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ%о
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall2^conv2d_6/kernel/Regularizer/Square/ReadVariableOp!^conv2d_7/StatefulPartitionedCall2^conv2d_7/kernel/Regularizer/Square/ReadVariableOp!^conv2d_8/StatefulPartitionedCall2^conv2d_8/kernel/Regularizer/Square/ReadVariableOp ^dense_2/StatefulPartitionedCall1^dense_2/kernel/Regularizer/Square/ReadVariableOp.^layer_normalization_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:џџџџџџџџџx': : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2f
1conv2d_6/kernel/Regularizer/Square/ReadVariableOp1conv2d_6/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2f
1conv2d_7/kernel/Regularizer/Square/ReadVariableOp1conv2d_7/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2f
1conv2d_8/kernel/Regularizer/Square/ReadVariableOp1conv2d_8/kernel/Regularizer/Square/ReadVariableOp2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2d
0dense_2/kernel/Regularizer/Square/ReadVariableOp0dense_2/kernel/Regularizer/Square/ReadVariableOp2^
-layer_normalization_1/StatefulPartitionedCall-layer_normalization_1/StatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџx'
 
_user_specified_nameinputs

K
/__inference_max_pooling2d_2_layer_call_fn_10116

inputs
identityм
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_7977
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
щ
g
K__inference_gaussian_noise_6_layer_call_and_return_conditional_losses_10145

inputs
identityV
IdentityIdentityinputs*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

f
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_10131

inputs
identity
MaxPoolMaxPoolinputs*/
_output_shapes
:џџџџџџџџџ*
data_formatNCHW*
ksize
*
paddingVALID*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ў
d
H__inference_activation_11_layer_call_and_return_conditional_losses_10613

inputs
identityN
SoftplusSoftplusinputs*
T0*'
_output_shapes
:џџџџџџџџџ%^
IdentityIdentitySoftplus:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ%"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ%:O K
'
_output_shapes
:џџџџџџџџџ%
 
_user_specified_nameinputs
м&
О
N__inference_batch_normalization_layer_call_and_return_conditional_losses_10278

inputsM
>assignmovingavg_readvariableop_batch_normalization_moving_mean:	аS
Dassignmovingavg_1_readvariableop_batch_normalization_moving_variance:	аE
6batchnorm_mul_readvariableop_batch_normalization_gamma:	а@
1batchnorm_readvariableop_batch_normalization_beta:	а
identityЂAssignMovingAvgЂAssignMovingAvg/ReadVariableOpЂAssignMovingAvg_1Ђ AssignMovingAvg_1/ReadVariableOpЂbatchnorm/ReadVariableOpЂbatchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	а*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	а
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:џџџџџџџџџаl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	а*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:а*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:а*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<
AssignMovingAvg/ReadVariableOpReadVariableOp>assignmovingavg_readvariableop_batch_normalization_moving_mean*
_output_shapes	
:а*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:аy
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:аУ
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
з#<Ђ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpDassignmovingavg_1_readvariableop_batch_normalization_moving_variance*
_output_shapes	
:а*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:а
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:аЯ
AssignMovingAvg_1AssignSubVariableOpDassignmovingavg_1_readvariableop_batch_normalization_moving_varianceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:аQ
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:а
batchnorm/mul/ReadVariableOpReadVariableOp6batchnorm_mul_readvariableop_batch_normalization_gamma*
_output_shapes	
:а*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:аd
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:џџџџџџџџџаi
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:а
batchnorm/ReadVariableOpReadVariableOp1batchnorm_readvariableop_batch_normalization_beta*
_output_shapes	
:а*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:аs
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:џџџџџџџџџаc
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџаъ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџа: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџа
 
_user_specified_nameinputs


*__inference_CNN_2D_NORM_layer_call_fn_9656

inputs1
layer_normalization_1_gamma:x'0
layer_normalization_1_beta:x')
conv2d_6_kernel:		x
conv2d_6_bias:)
conv2d_7_kernel:
conv2d_7_bias:.
batch_normalization_moving_mean:	а2
#batch_normalization_moving_variance:	а(
batch_normalization_gamma:	а'
batch_normalization_beta:	а)
conv2d_8_kernel:
conv2d_8_bias:/
!batch_normalization_1_moving_mean:Z3
%batch_normalization_1_moving_variance:Z)
batch_normalization_1_gamma:Z(
batch_normalization_1_beta:Z/
!batch_normalization_2_moving_mean:Z3
%batch_normalization_2_moving_variance:Z)
batch_normalization_2_gamma:Z(
batch_normalization_2_beta:Z 
dense_2_kernel:Z%
dense_2_bias:%
identityЂStatefulPartitionedCallЅ
StatefulPartitionedCallStatefulPartitionedCallinputslayer_normalization_1_gammalayer_normalization_1_betaconv2d_6_kernelconv2d_6_biasconv2d_7_kernelconv2d_7_biasbatch_normalization_moving_mean#batch_normalization_moving_variancebatch_normalization_gammabatch_normalization_betaconv2d_8_kernelconv2d_8_bias!batch_normalization_1_moving_mean%batch_normalization_1_moving_variancebatch_normalization_1_gammabatch_normalization_1_beta!batch_normalization_2_moving_mean%batch_normalization_2_moving_variancebatch_normalization_2_gammabatch_normalization_2_betadense_2_kerneldense_2_bias*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ%*2
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_CNN_2D_NORM_layer_call_and_return_conditional_losses_9181o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ%`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*Z
_input_shapesI
G:џџџџџџџџџx': : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџx'
 
_user_specified_nameinputs
ў&
У
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_8308

inputsN
@assignmovingavg_readvariableop_batch_normalization_2_moving_mean:ZT
Fassignmovingavg_1_readvariableop_batch_normalization_2_moving_variance:ZF
8batchnorm_mul_readvariableop_batch_normalization_2_gamma:ZA
3batchnorm_readvariableop_batch_normalization_2_beta:Z
identityЂAssignMovingAvgЂAssignMovingAvg/ReadVariableOpЂAssignMovingAvg_1Ђ AssignMovingAvg_1/ReadVariableOpЂbatchnorm/ReadVariableOpЂbatchnorm/mul/ReadVariableOph
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

:Z
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:џџџџџџџџџZl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
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
з#<
AssignMovingAvg/ReadVariableOpReadVariableOp@assignmovingavg_readvariableop_batch_normalization_2_moving_mean*
_output_shapes
:Z*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:Zx
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:ZХ
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
з#<Ѓ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpFassignmovingavg_1_readvariableop_batch_normalization_2_moving_variance*
_output_shapes
:Z*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:Z~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:Zб
AssignMovingAvg_1AssignSubVariableOpFassignmovingavg_1_readvariableop_batch_normalization_2_moving_varianceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:ZP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:Z
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
:џџџџџџџџџZh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:Z
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
:џџџџџџџџџZb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџZъ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџZ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџZ
 
_user_specified_nameinputs

f
J__inference_gaussian_noise_8_layer_call_and_return_conditional_losses_8573

inputs
identityV
IdentityIdentityinputs*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ѕ
М
C__inference_conv2d_8_layer_call_and_return_conditional_losses_10361

inputs?
%conv2d_readvariableop_conv2d_8_kernel:2
$biasadd_readvariableop_conv2d_8_bias:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOpЂ1conv2d_8/kernel/Regularizer/Square/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOp%conv2d_readvariableop_conv2d_8_kernel*&
_output_shapes
:*
dtype0Б
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
data_formatNCHW*
paddingVALID*
strides
w
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_conv2d_8_bias*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
data_formatNCHW
1conv2d_8/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%conv2d_readvariableop_conv2d_8_kernel*&
_output_shapes
:*
dtype0
"conv2d_8/kernel/Regularizer/SquareSquare9conv2d_8/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
!conv2d_8/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
conv2d_8/kernel/Regularizer/SumSum&conv2d_8/kernel/Regularizer/Square:y:0*conv2d_8/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv2d_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:
conv2d_8/kernel/Regularizer/mulMul*conv2d_8/kernel/Regularizer/mul/x:output:0(conv2d_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџЋ
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp2^conv2d_8/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2f
1conv2d_8/kernel/Regularizer/Square/ReadVariableOp1conv2d_8/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ	
 
_user_specified_nameinputs

`
D__inference_flatten_3_layer_call_and_return_conditional_losses_10372

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџZ   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:џџџџџџџџџZX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџZ"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ў&
У
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_8180

inputsN
@assignmovingavg_readvariableop_batch_normalization_1_moving_mean:ZT
Fassignmovingavg_1_readvariableop_batch_normalization_1_moving_variance:ZF
8batchnorm_mul_readvariableop_batch_normalization_1_gamma:ZA
3batchnorm_readvariableop_batch_normalization_1_beta:Z
identityЂAssignMovingAvgЂAssignMovingAvg/ReadVariableOpЂAssignMovingAvg_1Ђ AssignMovingAvg_1/ReadVariableOpЂbatchnorm/ReadVariableOpЂbatchnorm/mul/ReadVariableOph
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

:Z
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:џџџџџџџџџZl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
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
з#<
AssignMovingAvg/ReadVariableOpReadVariableOp@assignmovingavg_readvariableop_batch_normalization_1_moving_mean*
_output_shapes
:Z*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:Zx
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:ZХ
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
з#<Ѓ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpFassignmovingavg_1_readvariableop_batch_normalization_1_moving_variance*
_output_shapes
:Z*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:Z~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:Zб
AssignMovingAvg_1AssignSubVariableOpFassignmovingavg_1_readvariableop_batch_normalization_1_moving_varianceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:ZP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:Z
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
:џџџџџџџџџZh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:Z
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
:џџџџџџџџџZb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџZъ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџZ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџZ
 
_user_specified_nameinputs
'
Н
M__inference_batch_normalization_layer_call_and_return_conditional_losses_8052

inputsM
>assignmovingavg_readvariableop_batch_normalization_moving_mean:	аS
Dassignmovingavg_1_readvariableop_batch_normalization_moving_variance:	аE
6batchnorm_mul_readvariableop_batch_normalization_gamma:	а@
1batchnorm_readvariableop_batch_normalization_beta:	а
identityЂAssignMovingAvgЂAssignMovingAvg/ReadVariableOpЂAssignMovingAvg_1Ђ AssignMovingAvg_1/ReadVariableOpЂbatchnorm/ReadVariableOpЂbatchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	а*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	а
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:џџџџџџџџџаl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	а*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:а*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:а*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<
AssignMovingAvg/ReadVariableOpReadVariableOp>assignmovingavg_readvariableop_batch_normalization_moving_mean*
_output_shapes	
:а*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:аy
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:аУ
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
з#<Ђ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpDassignmovingavg_1_readvariableop_batch_normalization_moving_variance*
_output_shapes	
:а*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:а
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:аЯ
AssignMovingAvg_1AssignSubVariableOpDassignmovingavg_1_readvariableop_batch_normalization_moving_varianceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:аQ
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:а
batchnorm/mul/ReadVariableOpReadVariableOp6batchnorm_mul_readvariableop_batch_normalization_gamma*
_output_shapes	
:а*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:аd
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:џџџџџџџџџаi
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:а
batchnorm/ReadVariableOpReadVariableOp1batchnorm_readvariableop_batch_normalization_beta*
_output_shapes	
:а*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:аs
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:џџџџџџџџџаc
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџаъ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџа: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџа
 
_user_specified_nameinputs
Ё
^
B__inference_reshape_layer_call_and_return_conditional_losses_10297

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
valueB:б
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
value	B :	Љ
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ	`
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ	"
identityIdentity:output:0*'
_input_shapes
:џџџџџџџџџа:P L
(
_output_shapes
:џџџџџџџџџа
 
_user_specified_nameinputs
№Х
ъ
F__inference_CNN_2D_NORM_layer_call_and_return_conditional_losses_10049

inputs^
Hlayer_normalization_1_reshape_readvariableop_layer_normalization_1_gamma:x'_
Ilayer_normalization_1_reshape_1_readvariableop_layer_normalization_1_beta:x'H
.conv2d_6_conv2d_readvariableop_conv2d_6_kernel:		x;
-conv2d_6_biasadd_readvariableop_conv2d_6_bias:H
.conv2d_7_conv2d_readvariableop_conv2d_7_kernel:;
-conv2d_7_biasadd_readvariableop_conv2d_7_bias:a
Rbatch_normalization_assignmovingavg_readvariableop_batch_normalization_moving_mean:	аg
Xbatch_normalization_assignmovingavg_1_readvariableop_batch_normalization_moving_variance:	аY
Jbatch_normalization_batchnorm_mul_readvariableop_batch_normalization_gamma:	аT
Ebatch_normalization_batchnorm_readvariableop_batch_normalization_beta:	аH
.conv2d_8_conv2d_readvariableop_conv2d_8_kernel:;
-conv2d_8_biasadd_readvariableop_conv2d_8_bias:d
Vbatch_normalization_1_assignmovingavg_readvariableop_batch_normalization_1_moving_mean:Zj
\batch_normalization_1_assignmovingavg_1_readvariableop_batch_normalization_1_moving_variance:Z\
Nbatch_normalization_1_batchnorm_mul_readvariableop_batch_normalization_1_gamma:ZW
Ibatch_normalization_1_batchnorm_readvariableop_batch_normalization_1_beta:Zd
Vbatch_normalization_2_assignmovingavg_readvariableop_batch_normalization_2_moving_mean:Zj
\batch_normalization_2_assignmovingavg_1_readvariableop_batch_normalization_2_moving_variance:Z\
Nbatch_normalization_2_batchnorm_mul_readvariableop_batch_normalization_2_gamma:ZW
Ibatch_normalization_2_batchnorm_readvariableop_batch_normalization_2_beta:Z>
,dense_2_matmul_readvariableop_dense_2_kernel:Z%9
+dense_2_biasadd_readvariableop_dense_2_bias:%
identityЂ#batch_normalization/AssignMovingAvgЂ2batch_normalization/AssignMovingAvg/ReadVariableOpЂ%batch_normalization/AssignMovingAvg_1Ђ4batch_normalization/AssignMovingAvg_1/ReadVariableOpЂ,batch_normalization/batchnorm/ReadVariableOpЂ0batch_normalization/batchnorm/mul/ReadVariableOpЂ%batch_normalization_1/AssignMovingAvgЂ4batch_normalization_1/AssignMovingAvg/ReadVariableOpЂ'batch_normalization_1/AssignMovingAvg_1Ђ6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpЂ.batch_normalization_1/batchnorm/ReadVariableOpЂ2batch_normalization_1/batchnorm/mul/ReadVariableOpЂ%batch_normalization_2/AssignMovingAvgЂ4batch_normalization_2/AssignMovingAvg/ReadVariableOpЂ'batch_normalization_2/AssignMovingAvg_1Ђ6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpЂ.batch_normalization_2/batchnorm/ReadVariableOpЂ2batch_normalization_2/batchnorm/mul/ReadVariableOpЂconv2d_6/BiasAdd/ReadVariableOpЂconv2d_6/Conv2D/ReadVariableOpЂ1conv2d_6/kernel/Regularizer/Square/ReadVariableOpЂconv2d_7/BiasAdd/ReadVariableOpЂconv2d_7/Conv2D/ReadVariableOpЂ1conv2d_7/kernel/Regularizer/Square/ReadVariableOpЂconv2d_8/BiasAdd/ReadVariableOpЂconv2d_8/Conv2D/ReadVariableOpЂ1conv2d_8/kernel/Regularizer/Square/ReadVariableOpЂdense_2/BiasAdd/ReadVariableOpЂdense_2/MatMul/ReadVariableOpЂ0dense_2/kernel/Regularizer/Square/ReadVariableOpЂ,layer_normalization_1/Reshape/ReadVariableOpЂ.layer_normalization_1/Reshape_1/ReadVariableOp
4layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         М
"layer_normalization_1/moments/meanMeaninputs=layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
	keep_dims(Ё
*layer_normalization_1/moments/StopGradientStopGradient+layer_normalization_1/moments/mean:output:0*
T0*/
_output_shapes
:џџџџџџџџџЛ
/layer_normalization_1/moments/SquaredDifferenceSquaredDifferenceinputs3layer_normalization_1/moments/StopGradient:output:0*
T0*/
_output_shapes
:џџџџџџџџџx'
8layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         ё
&layer_normalization_1/moments/varianceMean3layer_normalization_1/moments/SquaredDifference:z:0Alayer_normalization_1/moments/variance/reduction_indices:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
	keep_dims(Й
,layer_normalization_1/Reshape/ReadVariableOpReadVariableOpHlayer_normalization_1_reshape_readvariableop_layer_normalization_1_gamma*"
_output_shapes
:x'*
dtype0|
#layer_normalization_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   x      '   Н
layer_normalization_1/ReshapeReshape4layer_normalization_1/Reshape/ReadVariableOp:value:0,layer_normalization_1/Reshape/shape:output:0*
T0*&
_output_shapes
:x'М
.layer_normalization_1/Reshape_1/ReadVariableOpReadVariableOpIlayer_normalization_1_reshape_1_readvariableop_layer_normalization_1_beta*"
_output_shapes
:x'*
dtype0~
%layer_normalization_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   x      '   У
layer_normalization_1/Reshape_1Reshape6layer_normalization_1/Reshape_1/ReadVariableOp:value:0.layer_normalization_1/Reshape_1/shape:output:0*
T0*&
_output_shapes
:x'j
%layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Пж3Ч
#layer_normalization_1/batchnorm/addAddV2/layer_normalization_1/moments/variance:output:0.layer_normalization_1/batchnorm/add/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
%layer_normalization_1/batchnorm/RsqrtRsqrt'layer_normalization_1/batchnorm/add:z:0*
T0*/
_output_shapes
:џџџџџџџџџЗ
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0&layer_normalization_1/Reshape:output:0*
T0*/
_output_shapes
:џџџџџџџџџx'
%layer_normalization_1/batchnorm/mul_1Mulinputs'layer_normalization_1/batchnorm/mul:z:0*
T0*/
_output_shapes
:џџџџџџџџџx'М
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*/
_output_shapes
:џџџџџџџџџx'Й
#layer_normalization_1/batchnorm/subSub(layer_normalization_1/Reshape_1:output:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*/
_output_shapes
:џџџџџџџџџx'М
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*/
_output_shapes
:џџџџџџџџџx'
conv2d_6/Conv2D/ReadVariableOpReadVariableOp.conv2d_6_conv2d_readvariableop_conv2d_6_kernel*&
_output_shapes
:		x*
dtype0ц
conv2d_6/Conv2DConv2D)layer_normalization_1/batchnorm/add_1:z:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
data_formatNCHW*
paddingVALID*
strides

conv2d_6/BiasAdd/ReadVariableOpReadVariableOp-conv2d_6_biasadd_readvariableop_conv2d_6_bias*
_output_shapes
:*
dtype0Џ
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
data_formatNCHWС
max_pooling2d_2/MaxPoolMaxPoolconv2d_6/BiasAdd:output:0*/
_output_shapes
:џџџџџџџџџ*
data_formatNCHW*
ksize
*
paddingVALID*
strides
f
gaussian_noise_6/ShapeShape max_pooling2d_2/MaxPool:output:0*
T0*
_output_shapes
:h
#gaussian_noise_6/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    j
%gaussian_noise_6/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ=Г
3gaussian_noise_6/random_normal/RandomStandardNormalRandomStandardNormalgaussian_noise_6/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
dtype0б
"gaussian_noise_6/random_normal/mulMul<gaussian_noise_6/random_normal/RandomStandardNormal:output:0.gaussian_noise_6/random_normal/stddev:output:0*
T0*/
_output_shapes
:џџџџџџџџџЗ
gaussian_noise_6/random_normalAddV2&gaussian_noise_6/random_normal/mul:z:0,gaussian_noise_6/random_normal/mean:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
gaussian_noise_6/addAddV2 max_pooling2d_2/MaxPool:output:0"gaussian_noise_6/random_normal:z:0*
T0*/
_output_shapes
:џџџџџџџџџm
activation_8/ReluRelugaussian_noise_6/add:z:0*
T0*/
_output_shapes
:џџџџџџџџџ
conv2d_7/Conv2D/ReadVariableOpReadVariableOp.conv2d_7_conv2d_readvariableop_conv2d_7_kernel*&
_output_shapes
:*
dtype0м
conv2d_7/Conv2DConv2Dactivation_8/Relu:activations:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ	*
data_formatNCHW*
paddingVALID*
strides

conv2d_7/BiasAdd/ReadVariableOpReadVariableOp-conv2d_7_biasadd_readvariableop_conv2d_7_bias*
_output_shapes
:*
dtype0Џ
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ	*
data_formatNCHW`
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџа  
flatten_2/ReshapeReshapeconv2d_7/BiasAdd:output:0flatten_2/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџа|
2batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: М
 batch_normalization/moments/meanMeanflatten_2/Reshape:output:0;batch_normalization/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	а*
	keep_dims(
(batch_normalization/moments/StopGradientStopGradient)batch_normalization/moments/mean:output:0*
T0*
_output_shapes
:	аФ
-batch_normalization/moments/SquaredDifferenceSquaredDifferenceflatten_2/Reshape:output:01batch_normalization/moments/StopGradient:output:0*
T0*(
_output_shapes
:џџџџџџџџџа
6batch_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: л
$batch_normalization/moments/varianceMean1batch_normalization/moments/SquaredDifference:z:0?batch_normalization/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	а*
	keep_dims(
#batch_normalization/moments/SqueezeSqueeze)batch_normalization/moments/mean:output:0*
T0*
_output_shapes	
:а*
squeeze_dims
 
%batch_normalization/moments/Squeeze_1Squeeze-batch_normalization/moments/variance:output:0*
T0*
_output_shapes	
:а*
squeeze_dims
 n
)batch_normalization/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<Т
2batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOpRbatch_normalization_assignmovingavg_readvariableop_batch_normalization_moving_mean*
_output_shapes	
:а*
dtype0О
'batch_normalization/AssignMovingAvg/subSub:batch_normalization/AssignMovingAvg/ReadVariableOp:value:0,batch_normalization/moments/Squeeze:output:0*
T0*
_output_shapes	
:аЕ
'batch_normalization/AssignMovingAvg/mulMul+batch_normalization/AssignMovingAvg/sub:z:02batch_normalization/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:а
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
з#<Ъ
4batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOpXbatch_normalization_assignmovingavg_1_readvariableop_batch_normalization_moving_variance*
_output_shapes	
:а*
dtype0Ф
)batch_normalization/AssignMovingAvg_1/subSub<batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:0.batch_normalization/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:аЛ
)batch_normalization/AssignMovingAvg_1/mulMul-batch_normalization/AssignMovingAvg_1/sub:z:04batch_normalization/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:а
%batch_normalization/AssignMovingAvg_1AssignSubVariableOpXbatch_normalization_assignmovingavg_1_readvariableop_batch_normalization_moving_variance-batch_normalization/AssignMovingAvg_1/mul:z:05^batch_normalization/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0h
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:Ў
!batch_normalization/batchnorm/addAddV2.batch_normalization/moments/Squeeze_1:output:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:аy
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:аИ
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOpJbatch_normalization_batchnorm_mul_readvariableop_batch_normalization_gamma*
_output_shapes	
:а*
dtype0Б
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:а 
#batch_normalization/batchnorm/mul_1Mulflatten_2/Reshape:output:0%batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:џџџџџџџџџаЅ
#batch_normalization/batchnorm/mul_2Mul,batch_normalization/moments/Squeeze:output:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:аЏ
,batch_normalization/batchnorm/ReadVariableOpReadVariableOpEbatch_normalization_batchnorm_readvariableop_batch_normalization_beta*
_output_shapes	
:а*
dtype0­
!batch_normalization/batchnorm/subSub4batch_normalization/batchnorm/ReadVariableOp:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:аЏ
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:џџџџџџџџџаd
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
valueB:љ
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
value	B :	б
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:
reshape/ReshapeReshape'batch_normalization/batchnorm/add_1:z:0reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ	^
gaussian_noise_7/ShapeShapereshape/Reshape:output:0*
T0*
_output_shapes
:h
#gaussian_noise_7/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    j
%gaussian_noise_7/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ=Г
3gaussian_noise_7/random_normal/RandomStandardNormalRandomStandardNormalgaussian_noise_7/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ	*
dtype0б
"gaussian_noise_7/random_normal/mulMul<gaussian_noise_7/random_normal/RandomStandardNormal:output:0.gaussian_noise_7/random_normal/stddev:output:0*
T0*/
_output_shapes
:џџџџџџџџџ	З
gaussian_noise_7/random_normalAddV2&gaussian_noise_7/random_normal/mul:z:0,gaussian_noise_7/random_normal/mean:output:0*
T0*/
_output_shapes
:џџџџџџџџџ	
gaussian_noise_7/addAddV2reshape/Reshape:output:0"gaussian_noise_7/random_normal:z:0*
T0*/
_output_shapes
:џџџџџџџџџ	m
activation_9/ReluRelugaussian_noise_7/add:z:0*
T0*/
_output_shapes
:џџџџџџџџџ	
conv2d_8/Conv2D/ReadVariableOpReadVariableOp.conv2d_8_conv2d_readvariableop_conv2d_8_kernel*&
_output_shapes
:*
dtype0м
conv2d_8/Conv2DConv2Dactivation_9/Relu:activations:0&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
data_formatNCHW*
paddingVALID*
strides

conv2d_8/BiasAdd/ReadVariableOpReadVariableOp-conv2d_8_biasadd_readvariableop_conv2d_8_bias*
_output_shapes
:*
dtype0Џ
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
data_formatNCHW`
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџZ   
flatten_3/ReshapeReshapeconv2d_8/BiasAdd:output:0flatten_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџZ~
4batch_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: П
"batch_normalization_1/moments/meanMeanflatten_3/Reshape:output:0=batch_normalization_1/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:Z*
	keep_dims(
*batch_normalization_1/moments/StopGradientStopGradient+batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes

:ZЧ
/batch_normalization_1/moments/SquaredDifferenceSquaredDifferenceflatten_3/Reshape:output:03batch_normalization_1/moments/StopGradient:output:0*
T0*'
_output_shapes
:џџџџџџџџџZ
8batch_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: р
&batch_normalization_1/moments/varianceMean3batch_normalization_1/moments/SquaredDifference:z:0Abatch_normalization_1/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:Z*
	keep_dims(
%batch_normalization_1/moments/SqueezeSqueeze+batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes
:Z*
squeeze_dims
 
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
з#<Ч
4batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOpVbatch_normalization_1_assignmovingavg_readvariableop_batch_normalization_1_moving_mean*
_output_shapes
:Z*
dtype0У
)batch_normalization_1/AssignMovingAvg/subSub<batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_1/moments/Squeeze:output:0*
T0*
_output_shapes
:ZК
)batch_normalization_1/AssignMovingAvg/mulMul-batch_normalization_1/AssignMovingAvg/sub:z:04batch_normalization_1/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:Z
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
з#<Я
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOp\batch_normalization_1_assignmovingavg_1_readvariableop_batch_normalization_1_moving_variance*
_output_shapes
:Z*
dtype0Щ
+batch_normalization_1/AssignMovingAvg_1/subSub>batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_1/moments/Squeeze_1:output:0*
T0*
_output_shapes
:ZР
+batch_normalization_1/AssignMovingAvg_1/mulMul/batch_normalization_1/AssignMovingAvg_1/sub:z:06batch_normalization_1/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:ZЉ
'batch_normalization_1/AssignMovingAvg_1AssignSubVariableOp\batch_normalization_1_assignmovingavg_1_readvariableop_batch_normalization_1_moving_variance/batch_normalization_1/AssignMovingAvg_1/mul:z:07^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0j
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:Г
#batch_normalization_1/batchnorm/addAddV20batch_normalization_1/moments/Squeeze_1:output:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes
:Z|
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
:ZН
2batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpNbatch_normalization_1_batchnorm_mul_readvariableop_batch_normalization_1_gamma*
_output_shapes
:Z*
dtype0Ж
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:0:batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:ZЃ
%batch_normalization_1/batchnorm/mul_1Mulflatten_3/Reshape:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџZЊ
%batch_normalization_1/batchnorm/mul_2Mul.batch_normalization_1/moments/Squeeze:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
:ZД
.batch_normalization_1/batchnorm/ReadVariableOpReadVariableOpIbatch_normalization_1_batchnorm_readvariableop_batch_normalization_1_beta*
_output_shapes
:Z*
dtype0В
#batch_normalization_1/batchnorm/subSub6batch_normalization_1/batchnorm/ReadVariableOp:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:ZД
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџZh
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
valueB:
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
value	B :л
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0"reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:Ѓ
reshape_1/ReshapeReshape)batch_normalization_1/batchnorm/add_1:z:0 reshape_1/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ`
gaussian_noise_8/ShapeShapereshape_1/Reshape:output:0*
T0*
_output_shapes
:h
#gaussian_noise_8/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    j
%gaussian_noise_8/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ=Г
3gaussian_noise_8/random_normal/RandomStandardNormalRandomStandardNormalgaussian_noise_8/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
dtype0б
"gaussian_noise_8/random_normal/mulMul<gaussian_noise_8/random_normal/RandomStandardNormal:output:0.gaussian_noise_8/random_normal/stddev:output:0*
T0*/
_output_shapes
:џџџџџџџџџЗ
gaussian_noise_8/random_normalAddV2&gaussian_noise_8/random_normal/mul:z:0,gaussian_noise_8/random_normal/mean:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
gaussian_noise_8/addAddV2reshape_1/Reshape:output:0"gaussian_noise_8/random_normal:z:0*
T0*/
_output_shapes
:џџџџџџџџџn
activation_10/ReluRelugaussian_noise_8/add:z:0*
T0*/
_output_shapes
:џџџџџџџџџ`
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџZ   
flatten_4/ReshapeReshape activation_10/Relu:activations:0flatten_4/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџZ~
4batch_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: П
"batch_normalization_2/moments/meanMeanflatten_4/Reshape:output:0=batch_normalization_2/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:Z*
	keep_dims(
*batch_normalization_2/moments/StopGradientStopGradient+batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes

:ZЧ
/batch_normalization_2/moments/SquaredDifferenceSquaredDifferenceflatten_4/Reshape:output:03batch_normalization_2/moments/StopGradient:output:0*
T0*'
_output_shapes
:џџџџџџџџџZ
8batch_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: р
&batch_normalization_2/moments/varianceMean3batch_normalization_2/moments/SquaredDifference:z:0Abatch_normalization_2/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:Z*
	keep_dims(
%batch_normalization_2/moments/SqueezeSqueeze+batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes
:Z*
squeeze_dims
 
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
з#<Ч
4batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOpVbatch_normalization_2_assignmovingavg_readvariableop_batch_normalization_2_moving_mean*
_output_shapes
:Z*
dtype0У
)batch_normalization_2/AssignMovingAvg/subSub<batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_2/moments/Squeeze:output:0*
T0*
_output_shapes
:ZК
)batch_normalization_2/AssignMovingAvg/mulMul-batch_normalization_2/AssignMovingAvg/sub:z:04batch_normalization_2/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:Z
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
з#<Я
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOp\batch_normalization_2_assignmovingavg_1_readvariableop_batch_normalization_2_moving_variance*
_output_shapes
:Z*
dtype0Щ
+batch_normalization_2/AssignMovingAvg_1/subSub>batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_2/moments/Squeeze_1:output:0*
T0*
_output_shapes
:ZР
+batch_normalization_2/AssignMovingAvg_1/mulMul/batch_normalization_2/AssignMovingAvg_1/sub:z:06batch_normalization_2/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:ZЉ
'batch_normalization_2/AssignMovingAvg_1AssignSubVariableOp\batch_normalization_2_assignmovingavg_1_readvariableop_batch_normalization_2_moving_variance/batch_normalization_2/AssignMovingAvg_1/mul:z:07^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0j
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:Г
#batch_normalization_2/batchnorm/addAddV20batch_normalization_2/moments/Squeeze_1:output:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:Z|
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:ZН
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpNbatch_normalization_2_batchnorm_mul_readvariableop_batch_normalization_2_gamma*
_output_shapes
:Z*
dtype0Ж
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:ZЃ
%batch_normalization_2/batchnorm/mul_1Mulflatten_4/Reshape:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџZЊ
%batch_normalization_2/batchnorm/mul_2Mul.batch_normalization_2/moments/Squeeze:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:ZД
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOpIbatch_normalization_2_batchnorm_readvariableop_batch_normalization_2_beta*
_output_shapes
:Z*
dtype0В
#batch_normalization_2/batchnorm/subSub6batch_normalization_2/batchnorm/ReadVariableOp:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:ZД
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџZ
dense_2/MatMul/ReadVariableOpReadVariableOp,dense_2_matmul_readvariableop_dense_2_kernel*
_output_shapes

:Z%*
dtype0
dense_2/MatMulMatMul)batch_normalization_2/batchnorm/add_1:z:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ%
dense_2/BiasAdd/ReadVariableOpReadVariableOp+dense_2_biasadd_readvariableop_dense_2_bias*
_output_shapes
:%*
dtype0
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ%r
dense_2/ActivityRegularizer/AbsAbsdense_2/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ%r
!dense_2/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_2/ActivityRegularizer/SumSum#dense_2/ActivityRegularizer/Abs:y:0*dense_2/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_2/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:
dense_2/ActivityRegularizer/mulMul*dense_2/ActivityRegularizer/mul/x:output:0(dense_2/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: i
!dense_2/ActivityRegularizer/ShapeShapedense_2/BiasAdd:output:0*
T0*
_output_shapes
:y
/dense_2/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1dense_2/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1dense_2/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:н
)dense_2/ActivityRegularizer/strided_sliceStridedSlice*dense_2/ActivityRegularizer/Shape:output:08dense_2/ActivityRegularizer/strided_slice/stack:output:0:dense_2/ActivityRegularizer/strided_slice/stack_1:output:0:dense_2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
 dense_2/ActivityRegularizer/CastCast2dense_2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 
#dense_2/ActivityRegularizer/truedivRealDiv#dense_2/ActivityRegularizer/mul:z:0$dense_2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: n
activation_11/SoftplusSoftplusdense_2/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ%Ј
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOp.conv2d_6_conv2d_readvariableop_conv2d_6_kernel*&
_output_shapes
:		x*
dtype0
"conv2d_6/kernel/Regularizer/SquareSquare9conv2d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		xz
!conv2d_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
conv2d_6/kernel/Regularizer/SumSum&conv2d_6/kernel/Regularizer/Square:y:0*conv2d_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0(conv2d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: Ј
1conv2d_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOp.conv2d_7_conv2d_readvariableop_conv2d_7_kernel*&
_output_shapes
:*
dtype0
"conv2d_7/kernel/Regularizer/SquareSquare9conv2d_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
!conv2d_7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
conv2d_7/kernel/Regularizer/SumSum&conv2d_7/kernel/Regularizer/Square:y:0*conv2d_7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv2d_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:
conv2d_7/kernel/Regularizer/mulMul*conv2d_7/kernel/Regularizer/mul/x:output:0(conv2d_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: Ј
1conv2d_8/kernel/Regularizer/Square/ReadVariableOpReadVariableOp.conv2d_8_conv2d_readvariableop_conv2d_8_kernel*&
_output_shapes
:*
dtype0
"conv2d_8/kernel/Regularizer/SquareSquare9conv2d_8/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
!conv2d_8/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
conv2d_8/kernel/Regularizer/SumSum&conv2d_8/kernel/Regularizer/Square:y:0*conv2d_8/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv2d_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:
conv2d_8/kernel/Regularizer/mulMul*conv2d_8/kernel/Regularizer/mul/x:output:0(conv2d_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp,dense_2_matmul_readvariableop_dense_2_kernel*
_output_shapes

:Z%*
dtype0
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:Z%q
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_2/kernel/Regularizer/SumSum%dense_2/kernel/Regularizer/Square:y:0)dense_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: s
IdentityIdentity$activation_11/Softplus:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ%ы
NoOpNoOp$^batch_normalization/AssignMovingAvg3^batch_normalization/AssignMovingAvg/ReadVariableOp&^batch_normalization/AssignMovingAvg_15^batch_normalization/AssignMovingAvg_1/ReadVariableOp-^batch_normalization/batchnorm/ReadVariableOp1^batch_normalization/batchnorm/mul/ReadVariableOp&^batch_normalization_1/AssignMovingAvg5^batch_normalization_1/AssignMovingAvg/ReadVariableOp(^batch_normalization_1/AssignMovingAvg_17^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp3^batch_normalization_1/batchnorm/mul/ReadVariableOp&^batch_normalization_2/AssignMovingAvg5^batch_normalization_2/AssignMovingAvg/ReadVariableOp(^batch_normalization_2/AssignMovingAvg_17^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_2/batchnorm/ReadVariableOp3^batch_normalization_2/batchnorm/mul/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp2^conv2d_6/kernel/Regularizer/Square/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp2^conv2d_7/kernel/Regularizer/Square/ReadVariableOp ^conv2d_8/BiasAdd/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp2^conv2d_8/kernel/Regularizer/Square/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp1^dense_2/kernel/Regularizer/Square/ReadVariableOp-^layer_normalization_1/Reshape/ReadVariableOp/^layer_normalization_1/Reshape_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*Z
_input_shapesI
G:џџџџџџџџџx': : : : : : : : : : : : : : : : : : : : : : 2J
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
2batch_normalization_2/batchnorm/mul/ReadVariableOp2batch_normalization_2/batchnorm/mul/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2f
1conv2d_6/kernel/Regularizer/Square/ReadVariableOp1conv2d_6/kernel/Regularizer/Square/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp2f
1conv2d_7/kernel/Regularizer/Square/ReadVariableOp1conv2d_7/kernel/Regularizer/Square/ReadVariableOp2B
conv2d_8/BiasAdd/ReadVariableOpconv2d_8/BiasAdd/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp2f
1conv2d_8/kernel/Regularizer/Square/ReadVariableOp1conv2d_8/kernel/Regularizer/Square/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2d
0dense_2/kernel/Regularizer/Square/ReadVariableOp0dense_2/kernel/Regularizer/Square/ReadVariableOp2\
,layer_normalization_1/Reshape/ReadVariableOp,layer_normalization_1/Reshape/ReadVariableOp2`
.layer_normalization_1/Reshape_1/ReadVariableOp.layer_normalization_1/Reshape_1/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџx'
 
_user_specified_nameinputs

Л
B__inference_conv2d_6_layer_call_and_return_conditional_losses_8435

inputs?
%conv2d_readvariableop_conv2d_6_kernel:		x2
$biasadd_readvariableop_conv2d_6_bias:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOpЂ1conv2d_6/kernel/Regularizer/Square/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOp%conv2d_readvariableop_conv2d_6_kernel*&
_output_shapes
:		x*
dtype0Б
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
data_formatNCHW*
paddingVALID*
strides
w
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_conv2d_6_bias*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
data_formatNCHW
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%conv2d_readvariableop_conv2d_6_kernel*&
_output_shapes
:		x*
dtype0
"conv2d_6/kernel/Regularizer/SquareSquare9conv2d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		xz
!conv2d_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
conv2d_6/kernel/Regularizer/SumSum&conv2d_6/kernel/Regularizer/Square:y:0*conv2d_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0(conv2d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџЋ
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp2^conv2d_6/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџx': : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2f
1conv2d_6/kernel/Regularizer/Square/ReadVariableOp1conv2d_6/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџx'
 
_user_specified_nameinputs

E
)__inference_flatten_3_layer_call_fn_10366

inputs
identityГ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџZ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_flatten_3_layer_call_and_return_conditional_losses_8546`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџZ"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
А	
Љ
5__inference_batch_normalization_2_layer_call_fn_10527

inputs/
!batch_normalization_2_moving_mean:Z3
%batch_normalization_2_moving_variance:Z)
batch_normalization_2_gamma:Z(
batch_normalization_2_beta:Z
identityЂStatefulPartitionedCallк
StatefulPartitionedCallStatefulPartitionedCallinputs!batch_normalization_2_moving_mean%batch_normalization_2_moving_variancebatch_normalization_2_gammabatch_normalization_2_beta*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџZ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_8308o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџZ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџZ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџZ
 
_user_specified_nameinputs

Д
__inference_loss_fn_3_10657Q
?dense_2_kernel_regularizer_square_readvariableop_dense_2_kernel:Z%
identityЂ0dense_2/kernel/Regularizer/Square/ReadVariableOpА
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp?dense_2_kernel_regularizer_square_readvariableop_dense_2_kernel*
_output_shapes

:Z%*
dtype0
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:Z%q
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_2/kernel/Regularizer/SumSum%dense_2/kernel/Regularizer/Square:y:0)dense_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: `
IdentityIdentity"dense_2/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: y
NoOpNoOp1^dense_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*
_input_shapes
: 2d
0dense_2/kernel/Regularizer/Square/ReadVariableOp0dense_2/kernel/Regularizer/Square/ReadVariableOp
К
П
__inference_loss_fn_0_10624[
Aconv2d_6_kernel_regularizer_square_readvariableop_conv2d_6_kernel:		x
identityЂ1conv2d_6/kernel/Regularizer/Square/ReadVariableOpЛ
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpAconv2d_6_kernel_regularizer_square_readvariableop_conv2d_6_kernel*&
_output_shapes
:		x*
dtype0
"conv2d_6/kernel/Regularizer/SquareSquare9conv2d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		xz
!conv2d_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
conv2d_6/kernel/Regularizer/SumSum&conv2d_6/kernel/Regularizer/Square:y:0*conv2d_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0(conv2d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentity#conv2d_6/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: z
NoOpNoOp2^conv2d_6/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*
_input_shapes
: 2f
1conv2d_6/kernel/Regularizer/Square/ReadVariableOp1conv2d_6/kernel/Regularizer/Square/ReadVariableOp
І	
i
J__inference_gaussian_noise_7_layer_call_and_return_conditional_losses_8883

inputs
identity;
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
 *ЭЬЬ=
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ	*
dtype0
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*/
_output_shapes
:џџџџџџџџџ	
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*/
_output_shapes
:џџџџџџџџџ	a
addAddV2inputsrandom_normal:z:0*
T0*/
_output_shapes
:џџџџџџџџџ	W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:џџџџџџџџџ	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ	:W S
/
_output_shapes
:џџџџџџџџџ	
 
_user_specified_nameinputs
Ќ

'__inference_dense_2_layer_call_fn_10594

inputs 
dense_2_kernel:Z%
dense_2_bias:%
identityЂStatefulPartitionedCallх
StatefulPartitionedCallStatefulPartitionedCallinputsdense_2_kerneldense_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ%*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_8611o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ%`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџZ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџZ
 
_user_specified_nameinputs
н

P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_10547

inputsL
>batchnorm_readvariableop_batch_normalization_2_moving_variance:ZF
8batchnorm_mul_readvariableop_batch_normalization_2_gamma:ZJ
<batchnorm_readvariableop_1_batch_normalization_2_moving_mean:ZC
5batchnorm_readvariableop_2_batch_normalization_2_beta:Z
identityЂbatchnorm/ReadVariableOpЂbatchnorm/ReadVariableOp_1Ђbatchnorm/ReadVariableOp_2Ђbatchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp>batchnorm_readvariableop_batch_normalization_2_moving_variance*
_output_shapes
:Z*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:ZP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:Z
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
:џџџџџџџџџZ
batchnorm/ReadVariableOp_1ReadVariableOp<batchnorm_readvariableop_1_batch_normalization_2_moving_mean*
_output_shapes
:Z*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:Z
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
:џџџџџџџџџZb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџZК
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџZ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџZ
 
_user_specified_nameinputs
Ќ
З
O__inference_layer_normalization_1_layer_call_and_return_conditional_losses_8415

inputsH
2reshape_readvariableop_layer_normalization_1_gamma:x'I
3reshape_1_readvariableop_layer_normalization_1_beta:x'
identityЂReshape/ReadVariableOpЂReshape_1/ReadVariableOps
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
	keep_dims(u
moments/StopGradientStopGradientmoments/mean:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*/
_output_shapes
:џџџџџџџџџx'w
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         Џ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
	keep_dims(
Reshape/ReadVariableOpReadVariableOp2reshape_readvariableop_layer_normalization_1_gamma*"
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
:x'
Reshape_1/ReadVariableOpReadVariableOp3reshape_1_readvariableop_layer_normalization_1_beta*"
_output_shapes
:x'*
dtype0h
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   x      '   
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:x'T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Пж3
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџe
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*/
_output_shapes
:џџџџџџџџџu
batchnorm/mulMulbatchnorm/Rsqrt:y:0Reshape:output:0*
T0*/
_output_shapes
:џџџџџџџџџx'k
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*/
_output_shapes
:џџџџџџџџџx'z
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*/
_output_shapes
:џџџџџџџџџx'w
batchnorm/subSubReshape_1:output:0batchnorm/mul_2:z:0*
T0*/
_output_shapes
:џџџџџџџџџx'z
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*/
_output_shapes
:џџџџџџџџџx'j
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџx'z
NoOpNoOp^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџx': : 20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџx'
 
_user_specified_nameinputs
с

E__inference_CNN_2D_NORM_layer_call_and_return_conditional_losses_9470
input_4G
1layer_normalization_1_layer_normalization_1_gamma:x'F
0layer_normalization_1_layer_normalization_1_beta:x'2
conv2d_6_conv2d_6_kernel:		x$
conv2d_6_conv2d_6_bias:2
conv2d_7_conv2d_7_kernel:$
conv2d_7_conv2d_7_bias:F
7batch_normalization_batch_normalization_moving_variance:	а<
-batch_normalization_batch_normalization_gamma:	аB
3batch_normalization_batch_normalization_moving_mean:	а;
,batch_normalization_batch_normalization_beta:	а2
conv2d_8_conv2d_8_kernel:$
conv2d_8_conv2d_8_bias:I
;batch_normalization_1_batch_normalization_1_moving_variance:Z?
1batch_normalization_1_batch_normalization_1_gamma:ZE
7batch_normalization_1_batch_normalization_1_moving_mean:Z>
0batch_normalization_1_batch_normalization_1_beta:ZI
;batch_normalization_2_batch_normalization_2_moving_variance:Z?
1batch_normalization_2_batch_normalization_2_gamma:ZE
7batch_normalization_2_batch_normalization_2_moving_mean:Z>
0batch_normalization_2_batch_normalization_2_beta:Z(
dense_2_dense_2_kernel:Z%"
dense_2_dense_2_bias:%
identityЂ+batch_normalization/StatefulPartitionedCallЂ-batch_normalization_1/StatefulPartitionedCallЂ-batch_normalization_2/StatefulPartitionedCallЂ conv2d_6/StatefulPartitionedCallЂ1conv2d_6/kernel/Regularizer/Square/ReadVariableOpЂ conv2d_7/StatefulPartitionedCallЂ1conv2d_7/kernel/Regularizer/Square/ReadVariableOpЂ conv2d_8/StatefulPartitionedCallЂ1conv2d_8/kernel/Regularizer/Square/ReadVariableOpЂdense_2/StatefulPartitionedCallЂ0dense_2/kernel/Regularizer/Square/ReadVariableOpЂ-layer_normalization_1/StatefulPartitionedCallй
-layer_normalization_1/StatefulPartitionedCallStatefulPartitionedCallinput_41layer_normalization_1_layer_normalization_1_gamma0layer_normalization_1_layer_normalization_1_beta*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџx'*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *X
fSRQ
O__inference_layer_normalization_1_layer_call_and_return_conditional_losses_8415Л
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_1/StatefulPartitionedCall:output:0conv2d_6_conv2d_6_kernelconv2d_6_conv2d_6_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_conv2d_6_layer_call_and_return_conditional_losses_8435є
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_8443ѕ
 gaussian_noise_6/PartitionedCallPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_gaussian_noise_6_layer_call_and_return_conditional_losses_8449ю
activation_8/PartitionedCallPartitionedCall)gaussian_noise_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_activation_8_layer_call_and_return_conditional_losses_8456Њ
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall%activation_8/PartitionedCall:output:0conv2d_7_conv2d_7_kernelconv2d_7_conv2d_7_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ	*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_conv2d_7_layer_call_and_return_conditional_losses_8474с
flatten_2/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџа* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_flatten_2_layer_call_and_return_conditional_losses_8484г
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:07batch_normalization_batch_normalization_moving_variance-batch_normalization_batch_normalization_gamma3batch_normalization_batch_normalization_moving_mean,batch_normalization_batch_normalization_beta*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџа*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_8009я
reshape/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ	* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_reshape_layer_call_and_return_conditional_losses_8505э
 gaussian_noise_7/PartitionedCallPartitionedCall reshape/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ	* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_gaussian_noise_7_layer_call_and_return_conditional_losses_8511ю
activation_9/PartitionedCallPartitionedCall)gaussian_noise_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ	* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_activation_9_layer_call_and_return_conditional_losses_8518Њ
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall%activation_9/PartitionedCall:output:0conv2d_8_conv2d_8_kernelconv2d_8_conv2d_8_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_conv2d_8_layer_call_and_return_conditional_losses_8536р
flatten_3/PartitionedCallPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџZ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_flatten_3_layer_call_and_return_conditional_losses_8546ц
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0;batch_normalization_1_batch_normalization_1_moving_variance1batch_normalization_1_batch_normalization_1_gamma7batch_normalization_1_batch_normalization_1_moving_mean0batch_normalization_1_batch_normalization_1_beta*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџZ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_8137ѕ
reshape_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_reshape_1_layer_call_and_return_conditional_losses_8567я
 gaussian_noise_8/PartitionedCallPartitionedCall"reshape_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_gaussian_noise_8_layer_call_and_return_conditional_losses_8573№
activation_10/PartitionedCallPartitionedCall)gaussian_noise_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_activation_10_layer_call_and_return_conditional_losses_8580н
flatten_4/PartitionedCallPartitionedCall&activation_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџZ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_flatten_4_layer_call_and_return_conditional_losses_8588ц
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0;batch_normalization_2_batch_normalization_2_moving_variance1batch_normalization_2_batch_normalization_2_gamma7batch_normalization_2_batch_normalization_2_moving_mean0batch_normalization_2_batch_normalization_2_beta*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџZ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_8265­
dense_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0dense_2_dense_2_kerneldense_2_dense_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ%*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_8611Ъ
+dense_2/ActivityRegularizer/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8 *6
f1R/
-__inference_dense_2_activity_regularizer_8626y
!dense_2/ActivityRegularizer/ShapeShape(dense_2/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:y
/dense_2/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1dense_2/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1dense_2/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:н
)dense_2/ActivityRegularizer/strided_sliceStridedSlice*dense_2/ActivityRegularizer/Shape:output:08dense_2/ActivityRegularizer/strided_slice/stack:output:0:dense_2/ActivityRegularizer/strided_slice/stack_1:output:0:dense_2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
 dense_2/ActivityRegularizer/CastCast2dense_2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: Ћ
#dense_2/ActivityRegularizer/truedivRealDiv4dense_2/ActivityRegularizer/PartitionedCall:output:0$dense_2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ч
activation_11/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ%* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_activation_11_layer_call_and_return_conditional_losses_8640
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_6_conv2d_6_kernel*&
_output_shapes
:		x*
dtype0
"conv2d_6/kernel/Regularizer/SquareSquare9conv2d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		xz
!conv2d_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
conv2d_6/kernel/Regularizer/SumSum&conv2d_6/kernel/Regularizer/Square:y:0*conv2d_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0(conv2d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
1conv2d_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_7_conv2d_7_kernel*&
_output_shapes
:*
dtype0
"conv2d_7/kernel/Regularizer/SquareSquare9conv2d_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
!conv2d_7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
conv2d_7/kernel/Regularizer/SumSum&conv2d_7/kernel/Regularizer/Square:y:0*conv2d_7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv2d_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:
conv2d_7/kernel/Regularizer/mulMul*conv2d_7/kernel/Regularizer/mul/x:output:0(conv2d_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
1conv2d_8/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_8_conv2d_8_kernel*&
_output_shapes
:*
dtype0
"conv2d_8/kernel/Regularizer/SquareSquare9conv2d_8/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
!conv2d_8/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
conv2d_8/kernel/Regularizer/SumSum&conv2d_8/kernel/Regularizer/Square:y:0*conv2d_8/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv2d_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:
conv2d_8/kernel/Regularizer/mulMul*conv2d_8/kernel/Regularizer/mul/x:output:0(conv2d_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_2_dense_2_kernel*
_output_shapes

:Z%*
dtype0
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:Z%q
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_2/kernel/Regularizer/SumSum%dense_2/kernel/Regularizer/Square:y:0)dense_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: u
IdentityIdentity&activation_11/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ%о
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall2^conv2d_6/kernel/Regularizer/Square/ReadVariableOp!^conv2d_7/StatefulPartitionedCall2^conv2d_7/kernel/Regularizer/Square/ReadVariableOp!^conv2d_8/StatefulPartitionedCall2^conv2d_8/kernel/Regularizer/Square/ReadVariableOp ^dense_2/StatefulPartitionedCall1^dense_2/kernel/Regularizer/Square/ReadVariableOp.^layer_normalization_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*Z
_input_shapesI
G:џџџџџџџџџx': : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2f
1conv2d_6/kernel/Regularizer/Square/ReadVariableOp1conv2d_6/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2f
1conv2d_7/kernel/Regularizer/Square/ReadVariableOp1conv2d_7/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2f
1conv2d_8/kernel/Regularizer/Square/ReadVariableOp1conv2d_8/kernel/Regularizer/Square/ReadVariableOp2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2d
0dense_2/kernel/Regularizer/Square/ReadVariableOp0dense_2/kernel/Regularizer/Square/ReadVariableOp2^
-layer_normalization_1/StatefulPartitionedCall-layer_normalization_1/StatefulPartitionedCall:X T
/
_output_shapes
:џџџџџџџџџx'
!
_user_specified_name	input_4
Њ
г
5__inference_layer_normalization_1_layer_call_fn_10056

inputs1
layer_normalization_1_gamma:x'0
layer_normalization_1_beta:x'
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputslayer_normalization_1_gammalayer_normalization_1_beta*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџx'*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *X
fSRQ
O__inference_layer_normalization_1_layer_call_and_return_conditional_losses_8415w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџx'`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџx': : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџx'
 
_user_specified_nameinputs
с
з
__inference__wrapped_model_7968
input_4j
Tcnn_2d_norm_layer_normalization_1_reshape_readvariableop_layer_normalization_1_gamma:x'k
Ucnn_2d_norm_layer_normalization_1_reshape_1_readvariableop_layer_normalization_1_beta:x'T
:cnn_2d_norm_conv2d_6_conv2d_readvariableop_conv2d_6_kernel:		xG
9cnn_2d_norm_conv2d_6_biasadd_readvariableop_conv2d_6_bias:T
:cnn_2d_norm_conv2d_7_conv2d_readvariableop_conv2d_7_kernel:G
9cnn_2d_norm_conv2d_7_biasadd_readvariableop_conv2d_7_bias:k
\cnn_2d_norm_batch_normalization_batchnorm_readvariableop_batch_normalization_moving_variance:	аe
Vcnn_2d_norm_batch_normalization_batchnorm_mul_readvariableop_batch_normalization_gamma:	аi
Zcnn_2d_norm_batch_normalization_batchnorm_readvariableop_1_batch_normalization_moving_mean:	аb
Scnn_2d_norm_batch_normalization_batchnorm_readvariableop_2_batch_normalization_beta:	аT
:cnn_2d_norm_conv2d_8_conv2d_readvariableop_conv2d_8_kernel:G
9cnn_2d_norm_conv2d_8_biasadd_readvariableop_conv2d_8_bias:n
`cnn_2d_norm_batch_normalization_1_batchnorm_readvariableop_batch_normalization_1_moving_variance:Zh
Zcnn_2d_norm_batch_normalization_1_batchnorm_mul_readvariableop_batch_normalization_1_gamma:Zl
^cnn_2d_norm_batch_normalization_1_batchnorm_readvariableop_1_batch_normalization_1_moving_mean:Ze
Wcnn_2d_norm_batch_normalization_1_batchnorm_readvariableop_2_batch_normalization_1_beta:Zn
`cnn_2d_norm_batch_normalization_2_batchnorm_readvariableop_batch_normalization_2_moving_variance:Zh
Zcnn_2d_norm_batch_normalization_2_batchnorm_mul_readvariableop_batch_normalization_2_gamma:Zl
^cnn_2d_norm_batch_normalization_2_batchnorm_readvariableop_1_batch_normalization_2_moving_mean:Ze
Wcnn_2d_norm_batch_normalization_2_batchnorm_readvariableop_2_batch_normalization_2_beta:ZJ
8cnn_2d_norm_dense_2_matmul_readvariableop_dense_2_kernel:Z%E
7cnn_2d_norm_dense_2_biasadd_readvariableop_dense_2_bias:%
identityЂ8CNN_2D_NORM/batch_normalization/batchnorm/ReadVariableOpЂ:CNN_2D_NORM/batch_normalization/batchnorm/ReadVariableOp_1Ђ:CNN_2D_NORM/batch_normalization/batchnorm/ReadVariableOp_2Ђ<CNN_2D_NORM/batch_normalization/batchnorm/mul/ReadVariableOpЂ:CNN_2D_NORM/batch_normalization_1/batchnorm/ReadVariableOpЂ<CNN_2D_NORM/batch_normalization_1/batchnorm/ReadVariableOp_1Ђ<CNN_2D_NORM/batch_normalization_1/batchnorm/ReadVariableOp_2Ђ>CNN_2D_NORM/batch_normalization_1/batchnorm/mul/ReadVariableOpЂ:CNN_2D_NORM/batch_normalization_2/batchnorm/ReadVariableOpЂ<CNN_2D_NORM/batch_normalization_2/batchnorm/ReadVariableOp_1Ђ<CNN_2D_NORM/batch_normalization_2/batchnorm/ReadVariableOp_2Ђ>CNN_2D_NORM/batch_normalization_2/batchnorm/mul/ReadVariableOpЂ+CNN_2D_NORM/conv2d_6/BiasAdd/ReadVariableOpЂ*CNN_2D_NORM/conv2d_6/Conv2D/ReadVariableOpЂ+CNN_2D_NORM/conv2d_7/BiasAdd/ReadVariableOpЂ*CNN_2D_NORM/conv2d_7/Conv2D/ReadVariableOpЂ+CNN_2D_NORM/conv2d_8/BiasAdd/ReadVariableOpЂ*CNN_2D_NORM/conv2d_8/Conv2D/ReadVariableOpЂ*CNN_2D_NORM/dense_2/BiasAdd/ReadVariableOpЂ)CNN_2D_NORM/dense_2/MatMul/ReadVariableOpЂ8CNN_2D_NORM/layer_normalization_1/Reshape/ReadVariableOpЂ:CNN_2D_NORM/layer_normalization_1/Reshape_1/ReadVariableOp
@CNN_2D_NORM/layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         е
.CNN_2D_NORM/layer_normalization_1/moments/meanMeaninput_4ICNN_2D_NORM/layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
	keep_dims(Й
6CNN_2D_NORM/layer_normalization_1/moments/StopGradientStopGradient7CNN_2D_NORM/layer_normalization_1/moments/mean:output:0*
T0*/
_output_shapes
:џџџџџџџџџд
;CNN_2D_NORM/layer_normalization_1/moments/SquaredDifferenceSquaredDifferenceinput_4?CNN_2D_NORM/layer_normalization_1/moments/StopGradient:output:0*
T0*/
_output_shapes
:џџџџџџџџџx'
DCNN_2D_NORM/layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         
2CNN_2D_NORM/layer_normalization_1/moments/varianceMean?CNN_2D_NORM/layer_normalization_1/moments/SquaredDifference:z:0MCNN_2D_NORM/layer_normalization_1/moments/variance/reduction_indices:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
	keep_dims(б
8CNN_2D_NORM/layer_normalization_1/Reshape/ReadVariableOpReadVariableOpTcnn_2d_norm_layer_normalization_1_reshape_readvariableop_layer_normalization_1_gamma*"
_output_shapes
:x'*
dtype0
/CNN_2D_NORM/layer_normalization_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   x      '   с
)CNN_2D_NORM/layer_normalization_1/ReshapeReshape@CNN_2D_NORM/layer_normalization_1/Reshape/ReadVariableOp:value:08CNN_2D_NORM/layer_normalization_1/Reshape/shape:output:0*
T0*&
_output_shapes
:x'д
:CNN_2D_NORM/layer_normalization_1/Reshape_1/ReadVariableOpReadVariableOpUcnn_2d_norm_layer_normalization_1_reshape_1_readvariableop_layer_normalization_1_beta*"
_output_shapes
:x'*
dtype0
1CNN_2D_NORM/layer_normalization_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   x      '   ч
+CNN_2D_NORM/layer_normalization_1/Reshape_1ReshapeBCNN_2D_NORM/layer_normalization_1/Reshape_1/ReadVariableOp:value:0:CNN_2D_NORM/layer_normalization_1/Reshape_1/shape:output:0*
T0*&
_output_shapes
:x'v
1CNN_2D_NORM/layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Пж3ы
/CNN_2D_NORM/layer_normalization_1/batchnorm/addAddV2;CNN_2D_NORM/layer_normalization_1/moments/variance:output:0:CNN_2D_NORM/layer_normalization_1/batchnorm/add/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџЉ
1CNN_2D_NORM/layer_normalization_1/batchnorm/RsqrtRsqrt3CNN_2D_NORM/layer_normalization_1/batchnorm/add:z:0*
T0*/
_output_shapes
:џџџџџџџџџл
/CNN_2D_NORM/layer_normalization_1/batchnorm/mulMul5CNN_2D_NORM/layer_normalization_1/batchnorm/Rsqrt:y:02CNN_2D_NORM/layer_normalization_1/Reshape:output:0*
T0*/
_output_shapes
:џџџџџџџџџx'А
1CNN_2D_NORM/layer_normalization_1/batchnorm/mul_1Mulinput_43CNN_2D_NORM/layer_normalization_1/batchnorm/mul:z:0*
T0*/
_output_shapes
:џџџџџџџџџx'р
1CNN_2D_NORM/layer_normalization_1/batchnorm/mul_2Mul7CNN_2D_NORM/layer_normalization_1/moments/mean:output:03CNN_2D_NORM/layer_normalization_1/batchnorm/mul:z:0*
T0*/
_output_shapes
:џџџџџџџџџx'н
/CNN_2D_NORM/layer_normalization_1/batchnorm/subSub4CNN_2D_NORM/layer_normalization_1/Reshape_1:output:05CNN_2D_NORM/layer_normalization_1/batchnorm/mul_2:z:0*
T0*/
_output_shapes
:џџџџџџџџџx'р
1CNN_2D_NORM/layer_normalization_1/batchnorm/add_1AddV25CNN_2D_NORM/layer_normalization_1/batchnorm/mul_1:z:03CNN_2D_NORM/layer_normalization_1/batchnorm/sub:z:0*
T0*/
_output_shapes
:џџџџџџџџџx'­
*CNN_2D_NORM/conv2d_6/Conv2D/ReadVariableOpReadVariableOp:cnn_2d_norm_conv2d_6_conv2d_readvariableop_conv2d_6_kernel*&
_output_shapes
:		x*
dtype0
CNN_2D_NORM/conv2d_6/Conv2DConv2D5CNN_2D_NORM/layer_normalization_1/batchnorm/add_1:z:02CNN_2D_NORM/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
data_formatNCHW*
paddingVALID*
strides
Ё
+CNN_2D_NORM/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp9cnn_2d_norm_conv2d_6_biasadd_readvariableop_conv2d_6_bias*
_output_shapes
:*
dtype0г
CNN_2D_NORM/conv2d_6/BiasAddBiasAdd$CNN_2D_NORM/conv2d_6/Conv2D:output:03CNN_2D_NORM/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
data_formatNCHWй
#CNN_2D_NORM/max_pooling2d_2/MaxPoolMaxPool%CNN_2D_NORM/conv2d_6/BiasAdd:output:0*/
_output_shapes
:џџџџџџџџџ*
data_formatNCHW*
ksize
*
paddingVALID*
strides

CNN_2D_NORM/activation_8/ReluRelu,CNN_2D_NORM/max_pooling2d_2/MaxPool:output:0*
T0*/
_output_shapes
:џџџџџџџџџ­
*CNN_2D_NORM/conv2d_7/Conv2D/ReadVariableOpReadVariableOp:cnn_2d_norm_conv2d_7_conv2d_readvariableop_conv2d_7_kernel*&
_output_shapes
:*
dtype0
CNN_2D_NORM/conv2d_7/Conv2DConv2D+CNN_2D_NORM/activation_8/Relu:activations:02CNN_2D_NORM/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ	*
data_formatNCHW*
paddingVALID*
strides
Ё
+CNN_2D_NORM/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp9cnn_2d_norm_conv2d_7_biasadd_readvariableop_conv2d_7_bias*
_output_shapes
:*
dtype0г
CNN_2D_NORM/conv2d_7/BiasAddBiasAdd$CNN_2D_NORM/conv2d_7/Conv2D:output:03CNN_2D_NORM/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ	*
data_formatNCHWl
CNN_2D_NORM/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџа  Ј
CNN_2D_NORM/flatten_2/ReshapeReshape%CNN_2D_NORM/conv2d_7/BiasAdd:output:0$CNN_2D_NORM/flatten_2/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџав
8CNN_2D_NORM/batch_normalization/batchnorm/ReadVariableOpReadVariableOp\cnn_2d_norm_batch_normalization_batchnorm_readvariableop_batch_normalization_moving_variance*
_output_shapes	
:а*
dtype0t
/CNN_2D_NORM/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:и
-CNN_2D_NORM/batch_normalization/batchnorm/addAddV2@CNN_2D_NORM/batch_normalization/batchnorm/ReadVariableOp:value:08CNN_2D_NORM/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:а
/CNN_2D_NORM/batch_normalization/batchnorm/RsqrtRsqrt1CNN_2D_NORM/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:аа
<CNN_2D_NORM/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOpVcnn_2d_norm_batch_normalization_batchnorm_mul_readvariableop_batch_normalization_gamma*
_output_shapes	
:а*
dtype0е
-CNN_2D_NORM/batch_normalization/batchnorm/mulMul3CNN_2D_NORM/batch_normalization/batchnorm/Rsqrt:y:0DCNN_2D_NORM/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:аФ
/CNN_2D_NORM/batch_normalization/batchnorm/mul_1Mul&CNN_2D_NORM/flatten_2/Reshape:output:01CNN_2D_NORM/batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:џџџџџџџџџав
:CNN_2D_NORM/batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOpZcnn_2d_norm_batch_normalization_batchnorm_readvariableop_1_batch_normalization_moving_mean*
_output_shapes	
:а*
dtype0г
/CNN_2D_NORM/batch_normalization/batchnorm/mul_2MulBCNN_2D_NORM/batch_normalization/batchnorm/ReadVariableOp_1:value:01CNN_2D_NORM/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:аЫ
:CNN_2D_NORM/batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOpScnn_2d_norm_batch_normalization_batchnorm_readvariableop_2_batch_normalization_beta*
_output_shapes	
:а*
dtype0г
-CNN_2D_NORM/batch_normalization/batchnorm/subSubBCNN_2D_NORM/batch_normalization/batchnorm/ReadVariableOp_2:value:03CNN_2D_NORM/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:аг
/CNN_2D_NORM/batch_normalization/batchnorm/add_1AddV23CNN_2D_NORM/batch_normalization/batchnorm/mul_1:z:01CNN_2D_NORM/batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:џџџџџџџџџа|
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
valueB:Е
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
value	B :	
!CNN_2D_NORM/reshape/Reshape/shapePack*CNN_2D_NORM/reshape/strided_slice:output:0,CNN_2D_NORM/reshape/Reshape/shape/1:output:0,CNN_2D_NORM/reshape/Reshape/shape/2:output:0,CNN_2D_NORM/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:С
CNN_2D_NORM/reshape/ReshapeReshape3CNN_2D_NORM/batch_normalization/batchnorm/add_1:z:0*CNN_2D_NORM/reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ	
CNN_2D_NORM/activation_9/ReluRelu$CNN_2D_NORM/reshape/Reshape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ	­
*CNN_2D_NORM/conv2d_8/Conv2D/ReadVariableOpReadVariableOp:cnn_2d_norm_conv2d_8_conv2d_readvariableop_conv2d_8_kernel*&
_output_shapes
:*
dtype0
CNN_2D_NORM/conv2d_8/Conv2DConv2D+CNN_2D_NORM/activation_9/Relu:activations:02CNN_2D_NORM/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
data_formatNCHW*
paddingVALID*
strides
Ё
+CNN_2D_NORM/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp9cnn_2d_norm_conv2d_8_biasadd_readvariableop_conv2d_8_bias*
_output_shapes
:*
dtype0г
CNN_2D_NORM/conv2d_8/BiasAddBiasAdd$CNN_2D_NORM/conv2d_8/Conv2D:output:03CNN_2D_NORM/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
data_formatNCHWl
CNN_2D_NORM/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџZ   Ї
CNN_2D_NORM/flatten_3/ReshapeReshape%CNN_2D_NORM/conv2d_8/BiasAdd:output:0$CNN_2D_NORM/flatten_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџZз
:CNN_2D_NORM/batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp`cnn_2d_norm_batch_normalization_1_batchnorm_readvariableop_batch_normalization_1_moving_variance*
_output_shapes
:Z*
dtype0v
1CNN_2D_NORM/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:н
/CNN_2D_NORM/batch_normalization_1/batchnorm/addAddV2BCNN_2D_NORM/batch_normalization_1/batchnorm/ReadVariableOp:value:0:CNN_2D_NORM/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes
:Z
1CNN_2D_NORM/batch_normalization_1/batchnorm/RsqrtRsqrt3CNN_2D_NORM/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
:Zе
>CNN_2D_NORM/batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpZcnn_2d_norm_batch_normalization_1_batchnorm_mul_readvariableop_batch_normalization_1_gamma*
_output_shapes
:Z*
dtype0к
/CNN_2D_NORM/batch_normalization_1/batchnorm/mulMul5CNN_2D_NORM/batch_normalization_1/batchnorm/Rsqrt:y:0FCNN_2D_NORM/batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:ZЧ
1CNN_2D_NORM/batch_normalization_1/batchnorm/mul_1Mul&CNN_2D_NORM/flatten_3/Reshape:output:03CNN_2D_NORM/batch_normalization_1/batchnorm/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџZз
<CNN_2D_NORM/batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOp^cnn_2d_norm_batch_normalization_1_batchnorm_readvariableop_1_batch_normalization_1_moving_mean*
_output_shapes
:Z*
dtype0и
1CNN_2D_NORM/batch_normalization_1/batchnorm/mul_2MulDCNN_2D_NORM/batch_normalization_1/batchnorm/ReadVariableOp_1:value:03CNN_2D_NORM/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
:Zа
<CNN_2D_NORM/batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOpWcnn_2d_norm_batch_normalization_1_batchnorm_readvariableop_2_batch_normalization_1_beta*
_output_shapes
:Z*
dtype0и
/CNN_2D_NORM/batch_normalization_1/batchnorm/subSubDCNN_2D_NORM/batch_normalization_1/batchnorm/ReadVariableOp_2:value:05CNN_2D_NORM/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Zи
1CNN_2D_NORM/batch_normalization_1/batchnorm/add_1AddV25CNN_2D_NORM/batch_normalization_1/batchnorm/mul_1:z:03CNN_2D_NORM/batch_normalization_1/batchnorm/sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџZ
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
valueB:П
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
value	B :
#CNN_2D_NORM/reshape_1/Reshape/shapePack,CNN_2D_NORM/reshape_1/strided_slice:output:0.CNN_2D_NORM/reshape_1/Reshape/shape/1:output:0.CNN_2D_NORM/reshape_1/Reshape/shape/2:output:0.CNN_2D_NORM/reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:Ч
CNN_2D_NORM/reshape_1/ReshapeReshape5CNN_2D_NORM/batch_normalization_1/batchnorm/add_1:z:0,CNN_2D_NORM/reshape_1/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
CNN_2D_NORM/activation_10/ReluRelu&CNN_2D_NORM/reshape_1/Reshape:output:0*
T0*/
_output_shapes
:џџџџџџџџџl
CNN_2D_NORM/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџZ   Ў
CNN_2D_NORM/flatten_4/ReshapeReshape,CNN_2D_NORM/activation_10/Relu:activations:0$CNN_2D_NORM/flatten_4/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџZз
:CNN_2D_NORM/batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp`cnn_2d_norm_batch_normalization_2_batchnorm_readvariableop_batch_normalization_2_moving_variance*
_output_shapes
:Z*
dtype0v
1CNN_2D_NORM/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:н
/CNN_2D_NORM/batch_normalization_2/batchnorm/addAddV2BCNN_2D_NORM/batch_normalization_2/batchnorm/ReadVariableOp:value:0:CNN_2D_NORM/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:Z
1CNN_2D_NORM/batch_normalization_2/batchnorm/RsqrtRsqrt3CNN_2D_NORM/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:Zе
>CNN_2D_NORM/batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpZcnn_2d_norm_batch_normalization_2_batchnorm_mul_readvariableop_batch_normalization_2_gamma*
_output_shapes
:Z*
dtype0к
/CNN_2D_NORM/batch_normalization_2/batchnorm/mulMul5CNN_2D_NORM/batch_normalization_2/batchnorm/Rsqrt:y:0FCNN_2D_NORM/batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:ZЧ
1CNN_2D_NORM/batch_normalization_2/batchnorm/mul_1Mul&CNN_2D_NORM/flatten_4/Reshape:output:03CNN_2D_NORM/batch_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџZз
<CNN_2D_NORM/batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOp^cnn_2d_norm_batch_normalization_2_batchnorm_readvariableop_1_batch_normalization_2_moving_mean*
_output_shapes
:Z*
dtype0и
1CNN_2D_NORM/batch_normalization_2/batchnorm/mul_2MulDCNN_2D_NORM/batch_normalization_2/batchnorm/ReadVariableOp_1:value:03CNN_2D_NORM/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:Zа
<CNN_2D_NORM/batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOpWcnn_2d_norm_batch_normalization_2_batchnorm_readvariableop_2_batch_normalization_2_beta*
_output_shapes
:Z*
dtype0и
/CNN_2D_NORM/batch_normalization_2/batchnorm/subSubDCNN_2D_NORM/batch_normalization_2/batchnorm/ReadVariableOp_2:value:05CNN_2D_NORM/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Zи
1CNN_2D_NORM/batch_normalization_2/batchnorm/add_1AddV25CNN_2D_NORM/batch_normalization_2/batchnorm/mul_1:z:03CNN_2D_NORM/batch_normalization_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџZЂ
)CNN_2D_NORM/dense_2/MatMul/ReadVariableOpReadVariableOp8cnn_2d_norm_dense_2_matmul_readvariableop_dense_2_kernel*
_output_shapes

:Z%*
dtype0Р
CNN_2D_NORM/dense_2/MatMulMatMul5CNN_2D_NORM/batch_normalization_2/batchnorm/add_1:z:01CNN_2D_NORM/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ%
*CNN_2D_NORM/dense_2/BiasAdd/ReadVariableOpReadVariableOp7cnn_2d_norm_dense_2_biasadd_readvariableop_dense_2_bias*
_output_shapes
:%*
dtype0В
CNN_2D_NORM/dense_2/BiasAddBiasAdd$CNN_2D_NORM/dense_2/MatMul:product:02CNN_2D_NORM/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ%
+CNN_2D_NORM/dense_2/ActivityRegularizer/AbsAbs$CNN_2D_NORM/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ%~
-CNN_2D_NORM/dense_2/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       М
+CNN_2D_NORM/dense_2/ActivityRegularizer/SumSum/CNN_2D_NORM/dense_2/ActivityRegularizer/Abs:y:06CNN_2D_NORM/dense_2/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: r
-CNN_2D_NORM/dense_2/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:С
+CNN_2D_NORM/dense_2/ActivityRegularizer/mulMul6CNN_2D_NORM/dense_2/ActivityRegularizer/mul/x:output:04CNN_2D_NORM/dense_2/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 
-CNN_2D_NORM/dense_2/ActivityRegularizer/ShapeShape$CNN_2D_NORM/dense_2/BiasAdd:output:0*
T0*
_output_shapes
:
;CNN_2D_NORM/dense_2/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
=CNN_2D_NORM/dense_2/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
=CNN_2D_NORM/dense_2/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
5CNN_2D_NORM/dense_2/ActivityRegularizer/strided_sliceStridedSlice6CNN_2D_NORM/dense_2/ActivityRegularizer/Shape:output:0DCNN_2D_NORM/dense_2/ActivityRegularizer/strided_slice/stack:output:0FCNN_2D_NORM/dense_2/ActivityRegularizer/strided_slice/stack_1:output:0FCNN_2D_NORM/dense_2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЄ
,CNN_2D_NORM/dense_2/ActivityRegularizer/CastCast>CNN_2D_NORM/dense_2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: О
/CNN_2D_NORM/dense_2/ActivityRegularizer/truedivRealDiv/CNN_2D_NORM/dense_2/ActivityRegularizer/mul:z:00CNN_2D_NORM/dense_2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 
"CNN_2D_NORM/activation_11/SoftplusSoftplus$CNN_2D_NORM/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ%
IdentityIdentity0CNN_2D_NORM/activation_11/Softplus:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ%

NoOpNoOp9^CNN_2D_NORM/batch_normalization/batchnorm/ReadVariableOp;^CNN_2D_NORM/batch_normalization/batchnorm/ReadVariableOp_1;^CNN_2D_NORM/batch_normalization/batchnorm/ReadVariableOp_2=^CNN_2D_NORM/batch_normalization/batchnorm/mul/ReadVariableOp;^CNN_2D_NORM/batch_normalization_1/batchnorm/ReadVariableOp=^CNN_2D_NORM/batch_normalization_1/batchnorm/ReadVariableOp_1=^CNN_2D_NORM/batch_normalization_1/batchnorm/ReadVariableOp_2?^CNN_2D_NORM/batch_normalization_1/batchnorm/mul/ReadVariableOp;^CNN_2D_NORM/batch_normalization_2/batchnorm/ReadVariableOp=^CNN_2D_NORM/batch_normalization_2/batchnorm/ReadVariableOp_1=^CNN_2D_NORM/batch_normalization_2/batchnorm/ReadVariableOp_2?^CNN_2D_NORM/batch_normalization_2/batchnorm/mul/ReadVariableOp,^CNN_2D_NORM/conv2d_6/BiasAdd/ReadVariableOp+^CNN_2D_NORM/conv2d_6/Conv2D/ReadVariableOp,^CNN_2D_NORM/conv2d_7/BiasAdd/ReadVariableOp+^CNN_2D_NORM/conv2d_7/Conv2D/ReadVariableOp,^CNN_2D_NORM/conv2d_8/BiasAdd/ReadVariableOp+^CNN_2D_NORM/conv2d_8/Conv2D/ReadVariableOp+^CNN_2D_NORM/dense_2/BiasAdd/ReadVariableOp*^CNN_2D_NORM/dense_2/MatMul/ReadVariableOp9^CNN_2D_NORM/layer_normalization_1/Reshape/ReadVariableOp;^CNN_2D_NORM/layer_normalization_1/Reshape_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*Z
_input_shapesI
G:џџџџџџџџџx': : : : : : : : : : : : : : : : : : : : : : 2t
8CNN_2D_NORM/batch_normalization/batchnorm/ReadVariableOp8CNN_2D_NORM/batch_normalization/batchnorm/ReadVariableOp2x
:CNN_2D_NORM/batch_normalization/batchnorm/ReadVariableOp_1:CNN_2D_NORM/batch_normalization/batchnorm/ReadVariableOp_12x
:CNN_2D_NORM/batch_normalization/batchnorm/ReadVariableOp_2:CNN_2D_NORM/batch_normalization/batchnorm/ReadVariableOp_22|
<CNN_2D_NORM/batch_normalization/batchnorm/mul/ReadVariableOp<CNN_2D_NORM/batch_normalization/batchnorm/mul/ReadVariableOp2x
:CNN_2D_NORM/batch_normalization_1/batchnorm/ReadVariableOp:CNN_2D_NORM/batch_normalization_1/batchnorm/ReadVariableOp2|
<CNN_2D_NORM/batch_normalization_1/batchnorm/ReadVariableOp_1<CNN_2D_NORM/batch_normalization_1/batchnorm/ReadVariableOp_12|
<CNN_2D_NORM/batch_normalization_1/batchnorm/ReadVariableOp_2<CNN_2D_NORM/batch_normalization_1/batchnorm/ReadVariableOp_22
>CNN_2D_NORM/batch_normalization_1/batchnorm/mul/ReadVariableOp>CNN_2D_NORM/batch_normalization_1/batchnorm/mul/ReadVariableOp2x
:CNN_2D_NORM/batch_normalization_2/batchnorm/ReadVariableOp:CNN_2D_NORM/batch_normalization_2/batchnorm/ReadVariableOp2|
<CNN_2D_NORM/batch_normalization_2/batchnorm/ReadVariableOp_1<CNN_2D_NORM/batch_normalization_2/batchnorm/ReadVariableOp_12|
<CNN_2D_NORM/batch_normalization_2/batchnorm/ReadVariableOp_2<CNN_2D_NORM/batch_normalization_2/batchnorm/ReadVariableOp_22
>CNN_2D_NORM/batch_normalization_2/batchnorm/mul/ReadVariableOp>CNN_2D_NORM/batch_normalization_2/batchnorm/mul/ReadVariableOp2Z
+CNN_2D_NORM/conv2d_6/BiasAdd/ReadVariableOp+CNN_2D_NORM/conv2d_6/BiasAdd/ReadVariableOp2X
*CNN_2D_NORM/conv2d_6/Conv2D/ReadVariableOp*CNN_2D_NORM/conv2d_6/Conv2D/ReadVariableOp2Z
+CNN_2D_NORM/conv2d_7/BiasAdd/ReadVariableOp+CNN_2D_NORM/conv2d_7/BiasAdd/ReadVariableOp2X
*CNN_2D_NORM/conv2d_7/Conv2D/ReadVariableOp*CNN_2D_NORM/conv2d_7/Conv2D/ReadVariableOp2Z
+CNN_2D_NORM/conv2d_8/BiasAdd/ReadVariableOp+CNN_2D_NORM/conv2d_8/BiasAdd/ReadVariableOp2X
*CNN_2D_NORM/conv2d_8/Conv2D/ReadVariableOp*CNN_2D_NORM/conv2d_8/Conv2D/ReadVariableOp2X
*CNN_2D_NORM/dense_2/BiasAdd/ReadVariableOp*CNN_2D_NORM/dense_2/BiasAdd/ReadVariableOp2V
)CNN_2D_NORM/dense_2/MatMul/ReadVariableOp)CNN_2D_NORM/dense_2/MatMul/ReadVariableOp2t
8CNN_2D_NORM/layer_normalization_1/Reshape/ReadVariableOp8CNN_2D_NORM/layer_normalization_1/Reshape/ReadVariableOp2x
:CNN_2D_NORM/layer_normalization_1/Reshape_1/ReadVariableOp:CNN_2D_NORM/layer_normalization_1/Reshape_1/ReadVariableOp:X T
/
_output_shapes
:џџџџџџџџџx'
!
_user_specified_name	input_4
џ
f
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_10126

inputs
identityЙ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
data_formatNCHW*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

E
)__inference_reshape_1_layer_call_fn_10449

inputs
identityЛ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_reshape_1_layer_call_and_return_conditional_losses_8567h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџZ:O K
'
_output_shapes
:џџџџџџџџџZ
 
_user_specified_nameinputs
сp
ш
!__inference__traced_restore_10874
file_prefixB
,assignvariableop_layer_normalization_1_gamma:x'C
-assignvariableop_1_layer_normalization_1_beta:x'<
"assignvariableop_2_conv2d_6_kernel:		x.
 assignvariableop_3_conv2d_6_bias:<
"assignvariableop_4_conv2d_7_kernel:.
 assignvariableop_5_conv2d_7_bias:;
,assignvariableop_6_batch_normalization_gamma:	а:
+assignvariableop_7_batch_normalization_beta:	аA
2assignvariableop_8_batch_normalization_moving_mean:	аE
6assignvariableop_9_batch_normalization_moving_variance:	а=
#assignvariableop_10_conv2d_8_kernel:/
!assignvariableop_11_conv2d_8_bias:=
/assignvariableop_12_batch_normalization_1_gamma:Z<
.assignvariableop_13_batch_normalization_1_beta:ZC
5assignvariableop_14_batch_normalization_1_moving_mean:ZG
9assignvariableop_15_batch_normalization_1_moving_variance:Z=
/assignvariableop_16_batch_normalization_2_gamma:Z<
.assignvariableop_17_batch_normalization_2_beta:ZC
5assignvariableop_18_batch_normalization_2_moving_mean:ZG
9assignvariableop_19_batch_normalization_2_moving_variance:Z4
"assignvariableop_20_dense_2_kernel:Z%.
 assignvariableop_21_dense_2_bias:%&
assignvariableop_22_total_12: &
assignvariableop_23_count_12: &
assignvariableop_24_total_13: &
assignvariableop_25_count_13: &
assignvariableop_26_total_14: &
assignvariableop_27_count_14: 
identity_29ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_3ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9З
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*н
valueгBаB5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЊ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*M
valueDBBB B B B B B B B B B B B B B B B B B B B B B B B B B B B B А
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapesv
t:::::::::::::::::::::::::::::*+
dtypes!
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp,assignvariableop_layer_normalization_1_gammaIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp-assignvariableop_1_layer_normalization_1_betaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_6_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_6_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv2d_7_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv2d_7_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp,assignvariableop_6_batch_normalization_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp+assignvariableop_7_batch_normalization_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:Ё
AssignVariableOp_8AssignVariableOp2assignvariableop_8_batch_normalization_moving_meanIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_9AssignVariableOp6assignvariableop_9_batch_normalization_moving_varianceIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp#assignvariableop_10_conv2d_8_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOp!assignvariableop_11_conv2d_8_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_12AssignVariableOp/assignvariableop_12_batch_normalization_1_gammaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp.assignvariableop_13_batch_normalization_1_betaIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_14AssignVariableOp5assignvariableop_14_batch_normalization_1_moving_meanIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:Њ
AssignVariableOp_15AssignVariableOp9assignvariableop_15_batch_normalization_1_moving_varianceIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_16AssignVariableOp/assignvariableop_16_batch_normalization_2_gammaIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp.assignvariableop_17_batch_normalization_2_betaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_18AssignVariableOp5assignvariableop_18_batch_normalization_2_moving_meanIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:Њ
AssignVariableOp_19AssignVariableOp9assignvariableop_19_batch_normalization_2_moving_varianceIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp"assignvariableop_20_dense_2_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp assignvariableop_21_dense_2_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOpassignvariableop_22_total_12Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOpassignvariableop_23_count_12Identity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOpassignvariableop_24_total_13Identity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOpassignvariableop_25_count_13Identity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOpassignvariableop_26_total_14Identity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOpassignvariableop_27_count_14Identity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 З
Identity_28Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_29IdentityIdentity_28:output:0^NoOp_1*
T0*
_output_shapes
: Є
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
е&
Ф
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_10444

inputsN
@assignmovingavg_readvariableop_batch_normalization_1_moving_mean:ZT
Fassignmovingavg_1_readvariableop_batch_normalization_1_moving_variance:ZF
8batchnorm_mul_readvariableop_batch_normalization_1_gamma:ZA
3batchnorm_readvariableop_batch_normalization_1_beta:Z
identityЂAssignMovingAvgЂAssignMovingAvg/ReadVariableOpЂAssignMovingAvg_1Ђ AssignMovingAvg_1/ReadVariableOpЂbatchnorm/ReadVariableOpЂbatchnorm/mul/ReadVariableOph
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

:Z
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:џџџџџџџџџZl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
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
з#<
AssignMovingAvg/ReadVariableOpReadVariableOp@assignmovingavg_readvariableop_batch_normalization_1_moving_mean*
_output_shapes
:Z*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:Zx
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:ZХ
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
з#<Ѓ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpFassignmovingavg_1_readvariableop_batch_normalization_1_moving_variance*
_output_shapes
:Z*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:Z~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:Zб
AssignMovingAvg_1AssignSubVariableOpFassignmovingavg_1_readvariableop_batch_normalization_1_moving_varianceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:ZP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:Z
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
:џџџџџџџџџZh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:Z
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
:џџџџџџџџџZb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџZъ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџZ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџZ
 
_user_specified_nameinputs
ь

"__inference_signature_wrapper_9602
input_41
layer_normalization_1_gamma:x'0
layer_normalization_1_beta:x')
conv2d_6_kernel:		x
conv2d_6_bias:)
conv2d_7_kernel:
conv2d_7_bias:2
#batch_normalization_moving_variance:	а(
batch_normalization_gamma:	а.
batch_normalization_moving_mean:	а'
batch_normalization_beta:	а)
conv2d_8_kernel:
conv2d_8_bias:3
%batch_normalization_1_moving_variance:Z)
batch_normalization_1_gamma:Z/
!batch_normalization_1_moving_mean:Z(
batch_normalization_1_beta:Z3
%batch_normalization_2_moving_variance:Z)
batch_normalization_2_gamma:Z/
!batch_normalization_2_moving_mean:Z(
batch_normalization_2_beta:Z 
dense_2_kernel:Z%
dense_2_bias:%
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_4layer_normalization_1_gammalayer_normalization_1_betaconv2d_6_kernelconv2d_6_biasconv2d_7_kernelconv2d_7_bias#batch_normalization_moving_variancebatch_normalization_gammabatch_normalization_moving_meanbatch_normalization_betaconv2d_8_kernelconv2d_8_bias%batch_normalization_1_moving_variancebatch_normalization_1_gamma!batch_normalization_1_moving_meanbatch_normalization_1_beta%batch_normalization_2_moving_variancebatch_normalization_2_gamma!batch_normalization_2_moving_meanbatch_normalization_2_betadense_2_kerneldense_2_bias*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ%*8
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 *(
f#R!
__inference__wrapped_model_7968o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ%`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*Z
_input_shapesI
G:џџџџџџџџџx': : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:џџџџџџџџџx'
!
_user_specified_name	input_4
§
j
K__inference_gaussian_noise_7_layer_call_and_return_conditional_losses_10322

inputs
identity;
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
 *ЭЬЬ=
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ	*
dtype0
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*/
_output_shapes
:џџџџџџџџџ	
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*/
_output_shapes
:џџџџџџџџџ	a
addAddV2inputsrandom_normal:z:0*
T0*/
_output_shapes
:џџџџџџџџџ	W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:џџџџџџџџџ	"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ	:W S
/
_output_shapes
:џџџџџџџџџ	
 
_user_specified_nameinputs
ї
i
0__inference_gaussian_noise_8_layer_call_fn_10473

inputs
identityЂStatefulPartitionedCallв
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_gaussian_noise_8_layer_call_and_return_conditional_losses_8779w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


*__inference_CNN_2D_NORM_layer_call_fn_9629

inputs1
layer_normalization_1_gamma:x'0
layer_normalization_1_beta:x')
conv2d_6_kernel:		x
conv2d_6_bias:)
conv2d_7_kernel:
conv2d_7_bias:2
#batch_normalization_moving_variance:	а(
batch_normalization_gamma:	а.
batch_normalization_moving_mean:	а'
batch_normalization_beta:	а)
conv2d_8_kernel:
conv2d_8_bias:3
%batch_normalization_1_moving_variance:Z)
batch_normalization_1_gamma:Z/
!batch_normalization_1_moving_mean:Z(
batch_normalization_1_beta:Z3
%batch_normalization_2_moving_variance:Z)
batch_normalization_2_gamma:Z/
!batch_normalization_2_moving_mean:Z(
batch_normalization_2_beta:Z 
dense_2_kernel:Z%
dense_2_bias:%
identityЂStatefulPartitionedCallЋ
StatefulPartitionedCallStatefulPartitionedCallinputslayer_normalization_1_gammalayer_normalization_1_betaconv2d_6_kernelconv2d_6_biasconv2d_7_kernelconv2d_7_bias#batch_normalization_moving_variancebatch_normalization_gammabatch_normalization_moving_meanbatch_normalization_betaconv2d_8_kernelconv2d_8_bias%batch_normalization_1_moving_variancebatch_normalization_1_gamma!batch_normalization_1_moving_meanbatch_normalization_1_beta%batch_normalization_2_moving_variancebatch_normalization_2_gamma!batch_normalization_2_moving_meanbatch_normalization_2_betadense_2_kerneldense_2_bias*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ%*8
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_CNN_2D_NORM_layer_call_and_return_conditional_losses_8667o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ%`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*Z
_input_shapesI
G:џџџџџџџџџx': : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџx'
 
_user_specified_nameinputs
ї
i
0__inference_gaussian_noise_6_layer_call_fn_10141

inputs
identityЂStatefulPartitionedCallв
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_gaussian_noise_6_layer_call_and_return_conditional_losses_8987w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
В	
Љ
5__inference_batch_normalization_2_layer_call_fn_10518

inputs3
%batch_normalization_2_moving_variance:Z)
batch_normalization_2_gamma:Z/
!batch_normalization_2_moving_mean:Z(
batch_normalization_2_beta:Z
identityЂStatefulPartitionedCallм
StatefulPartitionedCallStatefulPartitionedCallinputs%batch_normalization_2_moving_variancebatch_normalization_2_gamma!batch_normalization_2_moving_meanbatch_normalization_2_beta*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџZ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_8265o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџZ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџZ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџZ
 
_user_specified_nameinputs
§
j
K__inference_gaussian_noise_6_layer_call_and_return_conditional_losses_10156

inputs
identity;
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
 *ЭЬЬ=
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
dtype0
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*/
_output_shapes
:џџџџџџџџџa
addAddV2inputsrandom_normal:z:0*
T0*/
_output_shapes
:џџџџџџџџџW
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
К
П
__inference_loss_fn_2_10646[
Aconv2d_8_kernel_regularizer_square_readvariableop_conv2d_8_kernel:
identityЂ1conv2d_8/kernel/Regularizer/Square/ReadVariableOpЛ
1conv2d_8/kernel/Regularizer/Square/ReadVariableOpReadVariableOpAconv2d_8_kernel_regularizer_square_readvariableop_conv2d_8_kernel*&
_output_shapes
:*
dtype0
"conv2d_8/kernel/Regularizer/SquareSquare9conv2d_8/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
!conv2d_8/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
conv2d_8/kernel/Regularizer/SumSum&conv2d_8/kernel/Regularizer/Square:y:0*conv2d_8/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv2d_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:
conv2d_8/kernel/Regularizer/mulMul*conv2d_8/kernel/Regularizer/mul/x:output:0(conv2d_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentity#conv2d_8/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: z
NoOpNoOp2^conv2d_8/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*
_input_shapes
: 2f
1conv2d_8/kernel/Regularizer/Square/ReadVariableOp1conv2d_8/kernel/Regularizer/Square/ReadVariableOp

Л
B__inference_conv2d_8_layer_call_and_return_conditional_losses_8536

inputs?
%conv2d_readvariableop_conv2d_8_kernel:2
$biasadd_readvariableop_conv2d_8_bias:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOpЂ1conv2d_8/kernel/Regularizer/Square/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOp%conv2d_readvariableop_conv2d_8_kernel*&
_output_shapes
:*
dtype0Б
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
data_formatNCHW*
paddingVALID*
strides
w
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_conv2d_8_bias*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
data_formatNCHW
1conv2d_8/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%conv2d_readvariableop_conv2d_8_kernel*&
_output_shapes
:*
dtype0
"conv2d_8/kernel/Regularizer/SquareSquare9conv2d_8/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
!conv2d_8/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
conv2d_8/kernel/Regularizer/SumSum&conv2d_8/kernel/Regularizer/Square:y:0*conv2d_8/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv2d_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:
conv2d_8/kernel/Regularizer/mulMul*conv2d_8/kernel/Regularizer/mul/x:output:0(conv2d_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџЋ
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp2^conv2d_8/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2f
1conv2d_8/kernel/Regularizer/Square/ReadVariableOp1conv2d_8/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ	
 
_user_specified_nameinputs
§
j
K__inference_gaussian_noise_8_layer_call_and_return_conditional_losses_10488

inputs
identity;
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
 *ЭЬЬ=
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
dtype0
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*/
_output_shapes
:џџџџџџџџџa
addAddV2inputsrandom_normal:z:0*
T0*/
_output_shapes
:џџџџџџџџџW
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
щ
g
K__inference_gaussian_noise_7_layer_call_and_return_conditional_losses_10311

inputs
identityV
IdentityIdentityinputs*
T0*/
_output_shapes
:џџџџџџџџџ	"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ	:W S
/
_output_shapes
:џџџџџџџџџ	
 
_user_specified_nameinputs

Э
F__inference_dense_2_layer_call_and_return_all_conditional_losses_10603

inputs 
dense_2_kernel:Z%
dense_2_bias:%
identity

identity_1ЂStatefulPartitionedCallх
StatefulPartitionedCallStatefulPartitionedCallinputsdense_2_kerneldense_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ%*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_8611І
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
GPU2 *0J 8 *6
f1R/
-__inference_dense_2_activity_regularizer_8626o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ%X

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
:џџџџџџџџџZ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџZ
 
_user_specified_nameinputs
ѕ
М
C__inference_conv2d_6_layer_call_and_return_conditional_losses_10111

inputs?
%conv2d_readvariableop_conv2d_6_kernel:		x2
$biasadd_readvariableop_conv2d_6_bias:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOpЂ1conv2d_6/kernel/Regularizer/Square/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOp%conv2d_readvariableop_conv2d_6_kernel*&
_output_shapes
:		x*
dtype0Б
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
data_formatNCHW*
paddingVALID*
strides
w
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_conv2d_6_bias*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
data_formatNCHW
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%conv2d_readvariableop_conv2d_6_kernel*&
_output_shapes
:		x*
dtype0
"conv2d_6/kernel/Regularizer/SquareSquare9conv2d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		xz
!conv2d_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
conv2d_6/kernel/Regularizer/SumSum&conv2d_6/kernel/Regularizer/Square:y:0*conv2d_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0(conv2d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџЋ
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp2^conv2d_6/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџx': : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2f
1conv2d_6/kernel/Regularizer/Square/ReadVariableOp1conv2d_6/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџx'
 
_user_specified_nameinputs
В	
Љ
5__inference_batch_normalization_1_layer_call_fn_10381

inputs3
%batch_normalization_1_moving_variance:Z)
batch_normalization_1_gamma:Z/
!batch_normalization_1_moving_mean:Z(
batch_normalization_1_beta:Z
identityЂStatefulPartitionedCallм
StatefulPartitionedCallStatefulPartitionedCallinputs%batch_normalization_1_moving_variancebatch_normalization_1_gamma!batch_normalization_1_moving_meanbatch_normalization_1_beta*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџZ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_8137o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџZ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџZ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџZ
 
_user_specified_nameinputs
Т
d
H__inference_activation_10_layer_call_and_return_conditional_losses_10498

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:џџџџџџџџџb
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
У
_
C__inference_flatten_4_layer_call_and_return_conditional_losses_8588

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџZ   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:џџџџџџџџџZX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџZ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
з
c
G__inference_activation_11_layer_call_and_return_conditional_losses_8640

inputs
identityN
SoftplusSoftplusinputs*
T0*'
_output_shapes
:џџџџџџџџџ%^
IdentityIdentitySoftplus:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ%"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ%:O K
'
_output_shapes
:џџџџџџџџџ%
 
_user_specified_nameinputs"L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*И
serving_defaultЄ
C
input_48
serving_default_input_4:0џџџџџџџџџx'A
activation_110
StatefulPartitionedCall:0џџџџџџџџџ%tensorflow/serving/predict:вт
р
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
__call__
+&call_and_return_all_conditional_losses
_default_save_signature"
_tf_keras_network
"
_tf_keras_input_layer
Ц
axis
	gamma
beta
 	variables
!trainable_variables
"regularization_losses
#	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Н

$kernel
%bias
&	variables
'trainable_variables
(regularization_losses
)	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ї
*	variables
+trainable_variables
,regularization_losses
-	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ї
.	variables
/trainable_variables
0regularization_losses
1	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ї
2	variables
3trainable_variables
4regularization_losses
5	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Н

6kernel
7bias
8	variables
9trainable_variables
:regularization_losses
;	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ї
<	variables
=trainable_variables
>regularization_losses
?	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
ь
@axis
	Agamma
Bbeta
Cmoving_mean
Dmoving_variance
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ї
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ї
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
 __call__
+Ё&call_and_return_all_conditional_losses"
_tf_keras_layer
Ї
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
Ђ__call__
+Ѓ&call_and_return_all_conditional_losses"
_tf_keras_layer
Н

Ukernel
Vbias
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
Є__call__
+Ѕ&call_and_return_all_conditional_losses"
_tf_keras_layer
Ї
[	variables
\trainable_variables
]regularization_losses
^	keras_api
І__call__
+Ї&call_and_return_all_conditional_losses"
_tf_keras_layer
ь
_axis
	`gamma
abeta
bmoving_mean
cmoving_variance
d	variables
etrainable_variables
fregularization_losses
g	keras_api
Ј__call__
+Љ&call_and_return_all_conditional_losses"
_tf_keras_layer
Ї
h	variables
itrainable_variables
jregularization_losses
k	keras_api
Њ__call__
+Ћ&call_and_return_all_conditional_losses"
_tf_keras_layer
Ї
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
Ќ__call__
+­&call_and_return_all_conditional_losses"
_tf_keras_layer
Ї
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
Ў__call__
+Џ&call_and_return_all_conditional_losses"
_tf_keras_layer
Ї
t	variables
utrainable_variables
vregularization_losses
w	keras_api
А__call__
+Б&call_and_return_all_conditional_losses"
_tf_keras_layer
э
xaxis
	ygamma
zbeta
{moving_mean
|moving_variance
}	variables
~trainable_variables
regularization_losses
	keras_api
В__call__
+Г&call_and_return_all_conditional_losses"
_tf_keras_layer
У
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
Д__call__
+Е&call_and_return_all_conditional_losses"
_tf_keras_layer
Ћ
	variables
trainable_variables
regularization_losses
	keras_api
Ж__call__
+З&call_and_return_all_conditional_losses"
_tf_keras_layer
"
	optimizer
Ш
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
20
21"
trackable_list_wrapper

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
14
15"
trackable_list_wrapper
@
И0
Й1
К2
Л3"
trackable_list_wrapper
г
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
-
Мserving_default"
signature_map
 "
trackable_list_wrapper
1:/x'2layer_normalization_1/gamma
0:.x'2layer_normalization_1/beta
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
Е
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
 	variables
!trainable_variables
"regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
):'		x2conv2d_6/kernel
:2conv2d_6/bias
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
(
И0"
trackable_list_wrapper
Е
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
&	variables
'trainable_variables
(regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
*	variables
+trainable_variables
,regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
non_trainable_variables
 layers
Ёmetrics
 Ђlayer_regularization_losses
Ѓlayer_metrics
.	variables
/trainable_variables
0regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
Єnon_trainable_variables
Ѕlayers
Іmetrics
 Їlayer_regularization_losses
Јlayer_metrics
2	variables
3trainable_variables
4regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
):'2conv2d_7/kernel
:2conv2d_7/bias
.
60
71"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
(
Й0"
trackable_list_wrapper
Е
Љnon_trainable_variables
Њlayers
Ћmetrics
 Ќlayer_regularization_losses
­layer_metrics
8	variables
9trainable_variables
:regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
Ўnon_trainable_variables
Џlayers
Аmetrics
 Бlayer_regularization_losses
Вlayer_metrics
<	variables
=trainable_variables
>regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
(:&а2batch_normalization/gamma
':%а2batch_normalization/beta
0:.а (2batch_normalization/moving_mean
4:2а (2#batch_normalization/moving_variance
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
Е
Гnon_trainable_variables
Дlayers
Еmetrics
 Жlayer_regularization_losses
Зlayer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
Иnon_trainable_variables
Йlayers
Кmetrics
 Лlayer_regularization_losses
Мlayer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
Нnon_trainable_variables
Оlayers
Пmetrics
 Рlayer_regularization_losses
Сlayer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
 __call__
+Ё&call_and_return_all_conditional_losses
'Ё"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
Тnon_trainable_variables
Уlayers
Фmetrics
 Хlayer_regularization_losses
Цlayer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
Ђ__call__
+Ѓ&call_and_return_all_conditional_losses
'Ѓ"call_and_return_conditional_losses"
_generic_user_object
):'2conv2d_8/kernel
:2conv2d_8/bias
.
U0
V1"
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
(
К0"
trackable_list_wrapper
Е
Чnon_trainable_variables
Шlayers
Щmetrics
 Ъlayer_regularization_losses
Ыlayer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
Є__call__
+Ѕ&call_and_return_all_conditional_losses
'Ѕ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
Ьnon_trainable_variables
Эlayers
Юmetrics
 Яlayer_regularization_losses
аlayer_metrics
[	variables
\trainable_variables
]regularization_losses
І__call__
+Ї&call_and_return_all_conditional_losses
'Ї"call_and_return_conditional_losses"
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
Е
бnon_trainable_variables
вlayers
гmetrics
 дlayer_regularization_losses
еlayer_metrics
d	variables
etrainable_variables
fregularization_losses
Ј__call__
+Љ&call_and_return_all_conditional_losses
'Љ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
жnon_trainable_variables
зlayers
иmetrics
 йlayer_regularization_losses
кlayer_metrics
h	variables
itrainable_variables
jregularization_losses
Њ__call__
+Ћ&call_and_return_all_conditional_losses
'Ћ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
лnon_trainable_variables
мlayers
нmetrics
 оlayer_regularization_losses
пlayer_metrics
l	variables
mtrainable_variables
nregularization_losses
Ќ__call__
+­&call_and_return_all_conditional_losses
'­"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
рnon_trainable_variables
сlayers
тmetrics
 уlayer_regularization_losses
фlayer_metrics
p	variables
qtrainable_variables
rregularization_losses
Ў__call__
+Џ&call_and_return_all_conditional_losses
'Џ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
хnon_trainable_variables
цlayers
чmetrics
 шlayer_regularization_losses
щlayer_metrics
t	variables
utrainable_variables
vregularization_losses
А__call__
+Б&call_and_return_all_conditional_losses
'Б"call_and_return_conditional_losses"
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
Е
ъnon_trainable_variables
ыlayers
ьmetrics
 эlayer_regularization_losses
юlayer_metrics
}	variables
~trainable_variables
regularization_losses
В__call__
+Г&call_and_return_all_conditional_losses
'Г"call_and_return_conditional_losses"
_generic_user_object
 :Z%2dense_2/kernel
:%2dense_2/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
(
Л0"
trackable_list_wrapper
ж
яnon_trainable_variables
№layers
ёmetrics
 ђlayer_regularization_losses
ѓlayer_metrics
	variables
trainable_variables
regularization_losses
Д__call__
Нactivity_regularizer_fn
+Е&call_and_return_all_conditional_losses
'О"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
єnon_trainable_variables
ѕlayers
іmetrics
 їlayer_regularization_losses
јlayer_metrics
	variables
trainable_variables
regularization_losses
Ж__call__
+З&call_and_return_all_conditional_losses
'З"call_and_return_conditional_losses"
_generic_user_object
J
C0
D1
b2
c3
{4
|5"
trackable_list_wrapper
Ц
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
љ0
њ1
ћ2"
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
И0"
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
Й0"
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
К0"
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
Л0"
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

ќtotal

§count
ў
_fn_kwargs
џ	variables
	keras_api"
_tf_keras_metric
c

total

count

_fn_kwargs
	variables
	keras_api"
_tf_keras_metric
c

total

count

_fn_kwargs
	variables
	keras_api"
_tf_keras_metric
:  (2total_12
:  (2count_12
 "
trackable_dict_wrapper
0
ќ0
§1"
trackable_list_wrapper
.
џ	variables"
_generic_user_object
:  (2total_13
:  (2count_13
 "
trackable_dict_wrapper
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total_14
:  (2count_14
 "
trackable_dict_wrapper
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
і2ѓ
*__inference_CNN_2D_NORM_layer_call_fn_8692
*__inference_CNN_2D_NORM_layer_call_fn_9629
*__inference_CNN_2D_NORM_layer_call_fn_9656
*__inference_CNN_2D_NORM_layer_call_fn_9391Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
у2р
E__inference_CNN_2D_NORM_layer_call_and_return_conditional_losses_9821
F__inference_CNN_2D_NORM_layer_call_and_return_conditional_losses_10049
E__inference_CNN_2D_NORM_layer_call_and_return_conditional_losses_9470
E__inference_CNN_2D_NORM_layer_call_and_return_conditional_losses_9549Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ЪBЧ
__inference__wrapped_model_7968input_4"
В
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
п2м
5__inference_layer_normalization_1_layer_call_fn_10056Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
њ2ї
P__inference_layer_normalization_1_layer_call_and_return_conditional_losses_10082Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
в2Я
(__inference_conv2d_6_layer_call_fn_10095Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
э2ъ
C__inference_conv2d_6_layer_call_and_return_conditional_losses_10111Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2
/__inference_max_pooling2d_2_layer_call_fn_10116
/__inference_max_pooling2d_2_layer_call_fn_10121Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Р2Н
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_10126
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_10131Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2
0__inference_gaussian_noise_6_layer_call_fn_10136
0__inference_gaussian_noise_6_layer_call_fn_10141Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
д2б
K__inference_gaussian_noise_6_layer_call_and_return_conditional_losses_10145
K__inference_gaussian_noise_6_layer_call_and_return_conditional_losses_10156Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ж2г
,__inference_activation_8_layer_call_fn_10161Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ё2ю
G__inference_activation_8_layer_call_and_return_conditional_losses_10166Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
в2Я
(__inference_conv2d_7_layer_call_fn_10179Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
э2ъ
C__inference_conv2d_7_layer_call_and_return_conditional_losses_10195Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
г2а
)__inference_flatten_2_layer_call_fn_10200Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ю2ы
D__inference_flatten_2_layer_call_and_return_conditional_losses_10206Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Є2Ё
3__inference_batch_normalization_layer_call_fn_10215
3__inference_batch_normalization_layer_call_fn_10224Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
к2з
N__inference_batch_normalization_layer_call_and_return_conditional_losses_10244
N__inference_batch_normalization_layer_call_and_return_conditional_losses_10278Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
б2Ю
'__inference_reshape_layer_call_fn_10283Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ь2щ
B__inference_reshape_layer_call_and_return_conditional_losses_10297Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2
0__inference_gaussian_noise_7_layer_call_fn_10302
0__inference_gaussian_noise_7_layer_call_fn_10307Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
д2б
K__inference_gaussian_noise_7_layer_call_and_return_conditional_losses_10311
K__inference_gaussian_noise_7_layer_call_and_return_conditional_losses_10322Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ж2г
,__inference_activation_9_layer_call_fn_10327Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ё2ю
G__inference_activation_9_layer_call_and_return_conditional_losses_10332Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
в2Я
(__inference_conv2d_8_layer_call_fn_10345Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
э2ъ
C__inference_conv2d_8_layer_call_and_return_conditional_losses_10361Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
г2а
)__inference_flatten_3_layer_call_fn_10366Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ю2ы
D__inference_flatten_3_layer_call_and_return_conditional_losses_10372Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2Ѕ
5__inference_batch_normalization_1_layer_call_fn_10381
5__inference_batch_normalization_1_layer_call_fn_10390Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
о2л
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_10410
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_10444Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
г2а
)__inference_reshape_1_layer_call_fn_10449Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ю2ы
D__inference_reshape_1_layer_call_and_return_conditional_losses_10463Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2
0__inference_gaussian_noise_8_layer_call_fn_10468
0__inference_gaussian_noise_8_layer_call_fn_10473Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
д2б
K__inference_gaussian_noise_8_layer_call_and_return_conditional_losses_10477
K__inference_gaussian_noise_8_layer_call_and_return_conditional_losses_10488Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
з2д
-__inference_activation_10_layer_call_fn_10493Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ђ2я
H__inference_activation_10_layer_call_and_return_conditional_losses_10498Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
г2а
)__inference_flatten_4_layer_call_fn_10503Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ю2ы
D__inference_flatten_4_layer_call_and_return_conditional_losses_10509Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2Ѕ
5__inference_batch_normalization_2_layer_call_fn_10518
5__inference_batch_normalization_2_layer_call_fn_10527Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
о2л
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_10547
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_10581Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
б2Ю
'__inference_dense_2_layer_call_fn_10594Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
№2э
F__inference_dense_2_layer_call_and_return_all_conditional_losses_10603Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
з2д
-__inference_activation_11_layer_call_fn_10608Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ђ2я
H__inference_activation_11_layer_call_and_return_conditional_losses_10613Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
В2Џ
__inference_loss_fn_0_10624
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
В2Џ
__inference_loss_fn_1_10635
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
В2Џ
__inference_loss_fn_2_10646
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
В2Џ
__inference_loss_fn_3_10657
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
ЩBЦ
"__inference_signature_wrapper_9602input_4"
В
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
о2л
-__inference_dense_2_activity_regularizer_8382Љ
В
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ
	
ь2щ
B__inference_dense_2_layer_call_and_return_conditional_losses_10673Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 Э
F__inference_CNN_2D_NORM_layer_call_and_return_conditional_losses_10049$%67CDABUVbc`a{|yz?Ђ<
5Ђ2
(%
inputsџџџџџџџџџx'
p

 
Њ "%Ђ"

0џџџџџџџџџ%
 Э
E__inference_CNN_2D_NORM_layer_call_and_return_conditional_losses_9470$%67DACBUVc`ba|y{z@Ђ=
6Ђ3
)&
input_4џџџџџџџџџx'
p 

 
Њ "%Ђ"

0џџџџџџџџџ%
 Э
E__inference_CNN_2D_NORM_layer_call_and_return_conditional_losses_9549$%67CDABUVbc`a{|yz@Ђ=
6Ђ3
)&
input_4џџџџџџџџџx'
p

 
Њ "%Ђ"

0џџџџџџџџџ%
 Ь
E__inference_CNN_2D_NORM_layer_call_and_return_conditional_losses_9821$%67DACBUVc`ba|y{z?Ђ<
5Ђ2
(%
inputsџџџџџџџџџx'
p 

 
Њ "%Ђ"

0џџџџџџџџџ%
 Є
*__inference_CNN_2D_NORM_layer_call_fn_8692v$%67DACBUVc`ba|y{z@Ђ=
6Ђ3
)&
input_4џџџџџџџџџx'
p 

 
Њ "џџџџџџџџџ%Є
*__inference_CNN_2D_NORM_layer_call_fn_9391v$%67CDABUVbc`a{|yz@Ђ=
6Ђ3
)&
input_4џџџџџџџџџx'
p

 
Њ "џџџџџџџџџ%Ѓ
*__inference_CNN_2D_NORM_layer_call_fn_9629u$%67DACBUVc`ba|y{z?Ђ<
5Ђ2
(%
inputsџџџџџџџџџx'
p 

 
Њ "џџџџџџџџџ%Ѓ
*__inference_CNN_2D_NORM_layer_call_fn_9656u$%67CDABUVbc`a{|yz?Ђ<
5Ђ2
(%
inputsџџџџџџџџџx'
p

 
Њ "џџџџџџџџџ%З
__inference__wrapped_model_7968$%67DACBUVc`ba|y{z8Ђ5
.Ђ+
)&
input_4џџџџџџџџџx'
Њ "=Њ:
8
activation_11'$
activation_11џџџџџџџџџ%Д
H__inference_activation_10_layer_call_and_return_conditional_losses_10498h7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ "-Ђ*
# 
0џџџџџџџџџ
 
-__inference_activation_10_layer_call_fn_10493[7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ " џџџџџџџџџЄ
H__inference_activation_11_layer_call_and_return_conditional_losses_10613X/Ђ,
%Ђ"
 
inputsџџџџџџџџџ%
Њ "%Ђ"

0џџџџџџџџџ%
 |
-__inference_activation_11_layer_call_fn_10608K/Ђ,
%Ђ"
 
inputsџџџџџџџџџ%
Њ "џџџџџџџџџ%Г
G__inference_activation_8_layer_call_and_return_conditional_losses_10166h7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ "-Ђ*
# 
0џџџџџџџџџ
 
,__inference_activation_8_layer_call_fn_10161[7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ " џџџџџџџџџГ
G__inference_activation_9_layer_call_and_return_conditional_losses_10332h7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ	
Њ "-Ђ*
# 
0џџџџџџџџџ	
 
,__inference_activation_9_layer_call_fn_10327[7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ	
Њ " џџџџџџџџџ	Ж
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_10410bc`ba3Ђ0
)Ђ&
 
inputsџџџџџџџџџZ
p 
Њ "%Ђ"

0џџџџџџџџџZ
 Ж
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_10444bbc`a3Ђ0
)Ђ&
 
inputsџџџџџџџџџZ
p
Њ "%Ђ"

0џџџџџџџџџZ
 
5__inference_batch_normalization_1_layer_call_fn_10381Uc`ba3Ђ0
)Ђ&
 
inputsџџџџџџџџџZ
p 
Њ "џџџџџџџџџZ
5__inference_batch_normalization_1_layer_call_fn_10390Ubc`a3Ђ0
)Ђ&
 
inputsџџџџџџџџџZ
p
Њ "џџџџџџџџџZЖ
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_10547b|y{z3Ђ0
)Ђ&
 
inputsџџџџџџџџџZ
p 
Њ "%Ђ"

0џџџџџџџџџZ
 Ж
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_10581b{|yz3Ђ0
)Ђ&
 
inputsџџџџџџџџџZ
p
Њ "%Ђ"

0џџџџџџџџџZ
 
5__inference_batch_normalization_2_layer_call_fn_10518U|y{z3Ђ0
)Ђ&
 
inputsџџџџџџџџџZ
p 
Њ "џџџџџџџџџZ
5__inference_batch_normalization_2_layer_call_fn_10527U{|yz3Ђ0
)Ђ&
 
inputsџџџџџџџџџZ
p
Њ "џџџџџџџџџZЖ
N__inference_batch_normalization_layer_call_and_return_conditional_losses_10244dDACB4Ђ1
*Ђ'
!
inputsџџџџџџџџџа
p 
Њ "&Ђ#

0џџџџџџџџџа
 Ж
N__inference_batch_normalization_layer_call_and_return_conditional_losses_10278dCDAB4Ђ1
*Ђ'
!
inputsџџџџџџџџџа
p
Њ "&Ђ#

0џџџџџџџџџа
 
3__inference_batch_normalization_layer_call_fn_10215WDACB4Ђ1
*Ђ'
!
inputsџџџџџџџџџа
p 
Њ "џџџџџџџџџа
3__inference_batch_normalization_layer_call_fn_10224WCDAB4Ђ1
*Ђ'
!
inputsџџџџџџџџџа
p
Њ "џџџџџџџџџаГ
C__inference_conv2d_6_layer_call_and_return_conditional_losses_10111l$%7Ђ4
-Ђ*
(%
inputsџџџџџџџџџx'
Њ "-Ђ*
# 
0џџџџџџџџџ
 
(__inference_conv2d_6_layer_call_fn_10095_$%7Ђ4
-Ђ*
(%
inputsџџџџџџџџџx'
Њ " џџџџџџџџџГ
C__inference_conv2d_7_layer_call_and_return_conditional_losses_10195l677Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ "-Ђ*
# 
0џџџџџџџџџ	
 
(__inference_conv2d_7_layer_call_fn_10179_677Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ " џџџџџџџџџ	Г
C__inference_conv2d_8_layer_call_and_return_conditional_losses_10361lUV7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ	
Њ "-Ђ*
# 
0џџџџџџџџџ
 
(__inference_conv2d_8_layer_call_fn_10345_UV7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ	
Њ " џџџџџџџџџW
-__inference_dense_2_activity_regularizer_8382&Ђ
Ђ
	
x
Њ " Ж
F__inference_dense_2_layer_call_and_return_all_conditional_losses_10603l/Ђ,
%Ђ"
 
inputsџџџџџџџџџZ
Њ "3Ђ0

0џџџџџџџџџ%

	
1/0 Є
B__inference_dense_2_layer_call_and_return_conditional_losses_10673^/Ђ,
%Ђ"
 
inputsџџџџџџџџџZ
Њ "%Ђ"

0џџџџџџџџџ%
 |
'__inference_dense_2_layer_call_fn_10594Q/Ђ,
%Ђ"
 
inputsџџџџџџџџџZ
Њ "џџџџџџџџџ%Љ
D__inference_flatten_2_layer_call_and_return_conditional_losses_10206a7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ	
Њ "&Ђ#

0џџџџџџџџџа
 
)__inference_flatten_2_layer_call_fn_10200T7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ	
Њ "џџџџџџџџџаЈ
D__inference_flatten_3_layer_call_and_return_conditional_losses_10372`7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџZ
 
)__inference_flatten_3_layer_call_fn_10366S7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ "џџџџџџџџџZЈ
D__inference_flatten_4_layer_call_and_return_conditional_losses_10509`7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџZ
 
)__inference_flatten_4_layer_call_fn_10503S7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ "џџџџџџџџџZЛ
K__inference_gaussian_noise_6_layer_call_and_return_conditional_losses_10145l;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ
p 
Њ "-Ђ*
# 
0џџџџџџџџџ
 Л
K__inference_gaussian_noise_6_layer_call_and_return_conditional_losses_10156l;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ
p
Њ "-Ђ*
# 
0џџџџџџџџџ
 
0__inference_gaussian_noise_6_layer_call_fn_10136_;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ
p 
Њ " џџџџџџџџџ
0__inference_gaussian_noise_6_layer_call_fn_10141_;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ
p
Њ " џџџџџџџџџЛ
K__inference_gaussian_noise_7_layer_call_and_return_conditional_losses_10311l;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ	
p 
Њ "-Ђ*
# 
0џџџџџџџџџ	
 Л
K__inference_gaussian_noise_7_layer_call_and_return_conditional_losses_10322l;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ	
p
Њ "-Ђ*
# 
0џџџџџџџџџ	
 
0__inference_gaussian_noise_7_layer_call_fn_10302_;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ	
p 
Њ " џџџџџџџџџ	
0__inference_gaussian_noise_7_layer_call_fn_10307_;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ	
p
Њ " џџџџџџџџџ	Л
K__inference_gaussian_noise_8_layer_call_and_return_conditional_losses_10477l;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ
p 
Њ "-Ђ*
# 
0џџџџџџџџџ
 Л
K__inference_gaussian_noise_8_layer_call_and_return_conditional_losses_10488l;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ
p
Њ "-Ђ*
# 
0џџџџџџџџџ
 
0__inference_gaussian_noise_8_layer_call_fn_10468_;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ
p 
Њ " џџџџџџџџџ
0__inference_gaussian_noise_8_layer_call_fn_10473_;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ
p
Њ " џџџџџџџџџР
P__inference_layer_normalization_1_layer_call_and_return_conditional_losses_10082l7Ђ4
-Ђ*
(%
inputsџџџџџџџџџx'
Њ "-Ђ*
# 
0џџџџџџџџџx'
 
5__inference_layer_normalization_1_layer_call_fn_10056_7Ђ4
-Ђ*
(%
inputsџџџџџџџџџx'
Њ " џџџџџџџџџx':
__inference_loss_fn_0_10624$Ђ

Ђ 
Њ " :
__inference_loss_fn_1_106356Ђ

Ђ 
Њ " :
__inference_loss_fn_2_10646UЂ

Ђ 
Њ " ;
__inference_loss_fn_3_10657Ђ

Ђ 
Њ " э
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_10126RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ж
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_10131h7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ "-Ђ*
# 
0џџџџџџџџџ
 Х
/__inference_max_pooling2d_2_layer_call_fn_10116RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
/__inference_max_pooling2d_2_layer_call_fn_10121[7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ " џџџџџџџџџЈ
D__inference_reshape_1_layer_call_and_return_conditional_losses_10463`/Ђ,
%Ђ"
 
inputsџџџџџџџџџZ
Њ "-Ђ*
# 
0џџџџџџџџџ
 
)__inference_reshape_1_layer_call_fn_10449S/Ђ,
%Ђ"
 
inputsџџџџџџџџџZ
Њ " џџџџџџџџџЇ
B__inference_reshape_layer_call_and_return_conditional_losses_10297a0Ђ-
&Ђ#
!
inputsџџџџџџџџџа
Њ "-Ђ*
# 
0џџџџџџџџџ	
 
'__inference_reshape_layer_call_fn_10283T0Ђ-
&Ђ#
!
inputsџџџџџџџџџа
Њ " џџџџџџџџџ	Х
"__inference_signature_wrapper_9602$%67DACBUVc`ba|y{zCЂ@
Ђ 
9Њ6
4
input_4)&
input_4џџџџџџџџџx'"=Њ:
8
activation_11'$
activation_11џџџџџџџџџ%
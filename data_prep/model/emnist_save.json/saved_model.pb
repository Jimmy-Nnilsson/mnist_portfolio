??
??
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
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
?
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%??8"&
exponential_avg_factorfloat%  ??";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
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
?
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
?
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( ?
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
dtypetype?
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
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
?
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
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.11.02v2.11.0-rc2-15-g6290819256d8??
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
?
&Adam/v/module_wrapper_95/dense_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&Adam/v/module_wrapper_95/dense_19/bias
?
:Adam/v/module_wrapper_95/dense_19/bias/Read/ReadVariableOpReadVariableOp&Adam/v/module_wrapper_95/dense_19/bias*
_output_shapes
:*
dtype0
?
&Adam/m/module_wrapper_95/dense_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&Adam/m/module_wrapper_95/dense_19/bias
?
:Adam/m/module_wrapper_95/dense_19/bias/Read/ReadVariableOpReadVariableOp&Adam/m/module_wrapper_95/dense_19/bias*
_output_shapes
:*
dtype0
?
(Adam/v/module_wrapper_95/dense_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*9
shared_name*(Adam/v/module_wrapper_95/dense_19/kernel
?
<Adam/v/module_wrapper_95/dense_19/kernel/Read/ReadVariableOpReadVariableOp(Adam/v/module_wrapper_95/dense_19/kernel*
_output_shapes
:	?*
dtype0
?
(Adam/m/module_wrapper_95/dense_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*9
shared_name*(Adam/m/module_wrapper_95/dense_19/kernel
?
<Adam/m/module_wrapper_95/dense_19/kernel/Read/ReadVariableOpReadVariableOp(Adam/m/module_wrapper_95/dense_19/kernel*
_output_shapes
:	?*
dtype0
?
&Adam/v/module_wrapper_94/dense_18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&Adam/v/module_wrapper_94/dense_18/bias
?
:Adam/v/module_wrapper_94/dense_18/bias/Read/ReadVariableOpReadVariableOp&Adam/v/module_wrapper_94/dense_18/bias*
_output_shapes	
:?*
dtype0
?
&Adam/m/module_wrapper_94/dense_18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&Adam/m/module_wrapper_94/dense_18/bias
?
:Adam/m/module_wrapper_94/dense_18/bias/Read/ReadVariableOpReadVariableOp&Adam/m/module_wrapper_94/dense_18/bias*
_output_shapes	
:?*
dtype0
?
(Adam/v/module_wrapper_94/dense_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*9
shared_name*(Adam/v/module_wrapper_94/dense_18/kernel
?
<Adam/v/module_wrapper_94/dense_18/kernel/Read/ReadVariableOpReadVariableOp(Adam/v/module_wrapper_94/dense_18/kernel* 
_output_shapes
:
??*
dtype0
?
(Adam/m/module_wrapper_94/dense_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*9
shared_name*(Adam/m/module_wrapper_94/dense_18/kernel
?
<Adam/m/module_wrapper_94/dense_18/kernel/Read/ReadVariableOpReadVariableOp(Adam/m/module_wrapper_94/dense_18/kernel* 
_output_shapes
:
??*
dtype0
?
"Adam/v/batch_normalization_23/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Adam/v/batch_normalization_23/beta
?
6Adam/v/batch_normalization_23/beta/Read/ReadVariableOpReadVariableOp"Adam/v/batch_normalization_23/beta*
_output_shapes	
:?*
dtype0
?
"Adam/m/batch_normalization_23/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Adam/m/batch_normalization_23/beta
?
6Adam/m/batch_normalization_23/beta/Read/ReadVariableOpReadVariableOp"Adam/m/batch_normalization_23/beta*
_output_shapes	
:?*
dtype0
?
#Adam/v/batch_normalization_23/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#Adam/v/batch_normalization_23/gamma
?
7Adam/v/batch_normalization_23/gamma/Read/ReadVariableOpReadVariableOp#Adam/v/batch_normalization_23/gamma*
_output_shapes	
:?*
dtype0
?
#Adam/m/batch_normalization_23/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#Adam/m/batch_normalization_23/gamma
?
7Adam/m/batch_normalization_23/gamma/Read/ReadVariableOpReadVariableOp#Adam/m/batch_normalization_23/gamma*
_output_shapes	
:?*
dtype0
?
'Adam/v/module_wrapper_91/conv2d_45/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*8
shared_name)'Adam/v/module_wrapper_91/conv2d_45/bias
?
;Adam/v/module_wrapper_91/conv2d_45/bias/Read/ReadVariableOpReadVariableOp'Adam/v/module_wrapper_91/conv2d_45/bias*
_output_shapes	
:?*
dtype0
?
'Adam/m/module_wrapper_91/conv2d_45/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*8
shared_name)'Adam/m/module_wrapper_91/conv2d_45/bias
?
;Adam/m/module_wrapper_91/conv2d_45/bias/Read/ReadVariableOpReadVariableOp'Adam/m/module_wrapper_91/conv2d_45/bias*
_output_shapes	
:?*
dtype0
?
)Adam/v/module_wrapper_91/conv2d_45/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*:
shared_name+)Adam/v/module_wrapper_91/conv2d_45/kernel
?
=Adam/v/module_wrapper_91/conv2d_45/kernel/Read/ReadVariableOpReadVariableOp)Adam/v/module_wrapper_91/conv2d_45/kernel*(
_output_shapes
:??*
dtype0
?
)Adam/m/module_wrapper_91/conv2d_45/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*:
shared_name+)Adam/m/module_wrapper_91/conv2d_45/kernel
?
=Adam/m/module_wrapper_91/conv2d_45/kernel/Read/ReadVariableOpReadVariableOp)Adam/m/module_wrapper_91/conv2d_45/kernel*(
_output_shapes
:??*
dtype0
?
"Adam/v/batch_normalization_22/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Adam/v/batch_normalization_22/beta
?
6Adam/v/batch_normalization_22/beta/Read/ReadVariableOpReadVariableOp"Adam/v/batch_normalization_22/beta*
_output_shapes	
:?*
dtype0
?
"Adam/m/batch_normalization_22/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Adam/m/batch_normalization_22/beta
?
6Adam/m/batch_normalization_22/beta/Read/ReadVariableOpReadVariableOp"Adam/m/batch_normalization_22/beta*
_output_shapes	
:?*
dtype0
?
#Adam/v/batch_normalization_22/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#Adam/v/batch_normalization_22/gamma
?
7Adam/v/batch_normalization_22/gamma/Read/ReadVariableOpReadVariableOp#Adam/v/batch_normalization_22/gamma*
_output_shapes	
:?*
dtype0
?
#Adam/m/batch_normalization_22/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#Adam/m/batch_normalization_22/gamma
?
7Adam/m/batch_normalization_22/gamma/Read/ReadVariableOpReadVariableOp#Adam/m/batch_normalization_22/gamma*
_output_shapes	
:?*
dtype0
?
'Adam/v/module_wrapper_89/conv2d_44/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*8
shared_name)'Adam/v/module_wrapper_89/conv2d_44/bias
?
;Adam/v/module_wrapper_89/conv2d_44/bias/Read/ReadVariableOpReadVariableOp'Adam/v/module_wrapper_89/conv2d_44/bias*
_output_shapes	
:?*
dtype0
?
'Adam/m/module_wrapper_89/conv2d_44/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*8
shared_name)'Adam/m/module_wrapper_89/conv2d_44/bias
?
;Adam/m/module_wrapper_89/conv2d_44/bias/Read/ReadVariableOpReadVariableOp'Adam/m/module_wrapper_89/conv2d_44/bias*
_output_shapes	
:?*
dtype0
?
)Adam/v/module_wrapper_89/conv2d_44/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*:
shared_name+)Adam/v/module_wrapper_89/conv2d_44/kernel
?
=Adam/v/module_wrapper_89/conv2d_44/kernel/Read/ReadVariableOpReadVariableOp)Adam/v/module_wrapper_89/conv2d_44/kernel*(
_output_shapes
:??*
dtype0
?
)Adam/m/module_wrapper_89/conv2d_44/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*:
shared_name+)Adam/m/module_wrapper_89/conv2d_44/kernel
?
=Adam/m/module_wrapper_89/conv2d_44/kernel/Read/ReadVariableOpReadVariableOp)Adam/m/module_wrapper_89/conv2d_44/kernel*(
_output_shapes
:??*
dtype0
?
'Adam/v/module_wrapper_88/conv2d_43/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*8
shared_name)'Adam/v/module_wrapper_88/conv2d_43/bias
?
;Adam/v/module_wrapper_88/conv2d_43/bias/Read/ReadVariableOpReadVariableOp'Adam/v/module_wrapper_88/conv2d_43/bias*
_output_shapes	
:?*
dtype0
?
'Adam/m/module_wrapper_88/conv2d_43/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*8
shared_name)'Adam/m/module_wrapper_88/conv2d_43/bias
?
;Adam/m/module_wrapper_88/conv2d_43/bias/Read/ReadVariableOpReadVariableOp'Adam/m/module_wrapper_88/conv2d_43/bias*
_output_shapes	
:?*
dtype0
?
)Adam/v/module_wrapper_88/conv2d_43/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*:
shared_name+)Adam/v/module_wrapper_88/conv2d_43/kernel
?
=Adam/v/module_wrapper_88/conv2d_43/kernel/Read/ReadVariableOpReadVariableOp)Adam/v/module_wrapper_88/conv2d_43/kernel*'
_output_shapes
:@?*
dtype0
?
)Adam/m/module_wrapper_88/conv2d_43/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*:
shared_name+)Adam/m/module_wrapper_88/conv2d_43/kernel
?
=Adam/m/module_wrapper_88/conv2d_43/kernel/Read/ReadVariableOpReadVariableOp)Adam/m/module_wrapper_88/conv2d_43/kernel*'
_output_shapes
:@?*
dtype0
?
"Adam/v/batch_normalization_21/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/v/batch_normalization_21/beta
?
6Adam/v/batch_normalization_21/beta/Read/ReadVariableOpReadVariableOp"Adam/v/batch_normalization_21/beta*
_output_shapes
:@*
dtype0
?
"Adam/m/batch_normalization_21/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/m/batch_normalization_21/beta
?
6Adam/m/batch_normalization_21/beta/Read/ReadVariableOpReadVariableOp"Adam/m/batch_normalization_21/beta*
_output_shapes
:@*
dtype0
?
#Adam/v/batch_normalization_21/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/v/batch_normalization_21/gamma
?
7Adam/v/batch_normalization_21/gamma/Read/ReadVariableOpReadVariableOp#Adam/v/batch_normalization_21/gamma*
_output_shapes
:@*
dtype0
?
#Adam/m/batch_normalization_21/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/m/batch_normalization_21/gamma
?
7Adam/m/batch_normalization_21/gamma/Read/ReadVariableOpReadVariableOp#Adam/m/batch_normalization_21/gamma*
_output_shapes
:@*
dtype0
?
'Adam/v/module_wrapper_86/conv2d_42/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*8
shared_name)'Adam/v/module_wrapper_86/conv2d_42/bias
?
;Adam/v/module_wrapper_86/conv2d_42/bias/Read/ReadVariableOpReadVariableOp'Adam/v/module_wrapper_86/conv2d_42/bias*
_output_shapes
:@*
dtype0
?
'Adam/m/module_wrapper_86/conv2d_42/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*8
shared_name)'Adam/m/module_wrapper_86/conv2d_42/bias
?
;Adam/m/module_wrapper_86/conv2d_42/bias/Read/ReadVariableOpReadVariableOp'Adam/m/module_wrapper_86/conv2d_42/bias*
_output_shapes
:@*
dtype0
?
)Adam/v/module_wrapper_86/conv2d_42/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*:
shared_name+)Adam/v/module_wrapper_86/conv2d_42/kernel
?
=Adam/v/module_wrapper_86/conv2d_42/kernel/Read/ReadVariableOpReadVariableOp)Adam/v/module_wrapper_86/conv2d_42/kernel*&
_output_shapes
:@@*
dtype0
?
)Adam/m/module_wrapper_86/conv2d_42/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*:
shared_name+)Adam/m/module_wrapper_86/conv2d_42/kernel
?
=Adam/m/module_wrapper_86/conv2d_42/kernel/Read/ReadVariableOpReadVariableOp)Adam/m/module_wrapper_86/conv2d_42/kernel*&
_output_shapes
:@@*
dtype0
?
'Adam/v/module_wrapper_85/conv2d_41/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*8
shared_name)'Adam/v/module_wrapper_85/conv2d_41/bias
?
;Adam/v/module_wrapper_85/conv2d_41/bias/Read/ReadVariableOpReadVariableOp'Adam/v/module_wrapper_85/conv2d_41/bias*
_output_shapes
:@*
dtype0
?
'Adam/m/module_wrapper_85/conv2d_41/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*8
shared_name)'Adam/m/module_wrapper_85/conv2d_41/bias
?
;Adam/m/module_wrapper_85/conv2d_41/bias/Read/ReadVariableOpReadVariableOp'Adam/m/module_wrapper_85/conv2d_41/bias*
_output_shapes
:@*
dtype0
?
)Adam/v/module_wrapper_85/conv2d_41/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*:
shared_name+)Adam/v/module_wrapper_85/conv2d_41/kernel
?
=Adam/v/module_wrapper_85/conv2d_41/kernel/Read/ReadVariableOpReadVariableOp)Adam/v/module_wrapper_85/conv2d_41/kernel*&
_output_shapes
:@*
dtype0
?
)Adam/m/module_wrapper_85/conv2d_41/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*:
shared_name+)Adam/m/module_wrapper_85/conv2d_41/kernel
?
=Adam/m/module_wrapper_85/conv2d_41/kernel/Read/ReadVariableOpReadVariableOp)Adam/m/module_wrapper_85/conv2d_41/kernel*&
_output_shapes
:@*
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
?
module_wrapper_95/dense_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!module_wrapper_95/dense_19/bias
?
3module_wrapper_95/dense_19/bias/Read/ReadVariableOpReadVariableOpmodule_wrapper_95/dense_19/bias*
_output_shapes
:*
dtype0
?
!module_wrapper_95/dense_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*2
shared_name#!module_wrapper_95/dense_19/kernel
?
5module_wrapper_95/dense_19/kernel/Read/ReadVariableOpReadVariableOp!module_wrapper_95/dense_19/kernel*
_output_shapes
:	?*
dtype0
?
module_wrapper_94/dense_18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*0
shared_name!module_wrapper_94/dense_18/bias
?
3module_wrapper_94/dense_18/bias/Read/ReadVariableOpReadVariableOpmodule_wrapper_94/dense_18/bias*
_output_shapes	
:?*
dtype0
?
!module_wrapper_94/dense_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*2
shared_name#!module_wrapper_94/dense_18/kernel
?
5module_wrapper_94/dense_18/kernel/Read/ReadVariableOpReadVariableOp!module_wrapper_94/dense_18/kernel* 
_output_shapes
:
??*
dtype0
?
 module_wrapper_91/conv2d_45/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" module_wrapper_91/conv2d_45/bias
?
4module_wrapper_91/conv2d_45/bias/Read/ReadVariableOpReadVariableOp module_wrapper_91/conv2d_45/bias*
_output_shapes	
:?*
dtype0
?
"module_wrapper_91/conv2d_45/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*3
shared_name$"module_wrapper_91/conv2d_45/kernel
?
6module_wrapper_91/conv2d_45/kernel/Read/ReadVariableOpReadVariableOp"module_wrapper_91/conv2d_45/kernel*(
_output_shapes
:??*
dtype0
?
 module_wrapper_89/conv2d_44/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" module_wrapper_89/conv2d_44/bias
?
4module_wrapper_89/conv2d_44/bias/Read/ReadVariableOpReadVariableOp module_wrapper_89/conv2d_44/bias*
_output_shapes	
:?*
dtype0
?
"module_wrapper_89/conv2d_44/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*3
shared_name$"module_wrapper_89/conv2d_44/kernel
?
6module_wrapper_89/conv2d_44/kernel/Read/ReadVariableOpReadVariableOp"module_wrapper_89/conv2d_44/kernel*(
_output_shapes
:??*
dtype0
?
 module_wrapper_88/conv2d_43/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" module_wrapper_88/conv2d_43/bias
?
4module_wrapper_88/conv2d_43/bias/Read/ReadVariableOpReadVariableOp module_wrapper_88/conv2d_43/bias*
_output_shapes	
:?*
dtype0
?
"module_wrapper_88/conv2d_43/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*3
shared_name$"module_wrapper_88/conv2d_43/kernel
?
6module_wrapper_88/conv2d_43/kernel/Read/ReadVariableOpReadVariableOp"module_wrapper_88/conv2d_43/kernel*'
_output_shapes
:@?*
dtype0
?
 module_wrapper_86/conv2d_42/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" module_wrapper_86/conv2d_42/bias
?
4module_wrapper_86/conv2d_42/bias/Read/ReadVariableOpReadVariableOp module_wrapper_86/conv2d_42/bias*
_output_shapes
:@*
dtype0
?
"module_wrapper_86/conv2d_42/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*3
shared_name$"module_wrapper_86/conv2d_42/kernel
?
6module_wrapper_86/conv2d_42/kernel/Read/ReadVariableOpReadVariableOp"module_wrapper_86/conv2d_42/kernel*&
_output_shapes
:@@*
dtype0
?
 module_wrapper_85/conv2d_41/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" module_wrapper_85/conv2d_41/bias
?
4module_wrapper_85/conv2d_41/bias/Read/ReadVariableOpReadVariableOp module_wrapper_85/conv2d_41/bias*
_output_shapes
:@*
dtype0
?
"module_wrapper_85/conv2d_41/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"module_wrapper_85/conv2d_41/kernel
?
6module_wrapper_85/conv2d_41/kernel/Read/ReadVariableOpReadVariableOp"module_wrapper_85/conv2d_41/kernel*&
_output_shapes
:@*
dtype0
?
&batch_normalization_23/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&batch_normalization_23/moving_variance
?
:batch_normalization_23/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_23/moving_variance*
_output_shapes	
:?*
dtype0
?
"batch_normalization_23/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"batch_normalization_23/moving_mean
?
6batch_normalization_23/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_23/moving_mean*
_output_shapes	
:?*
dtype0
?
batch_normalization_23/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namebatch_normalization_23/beta
?
/batch_normalization_23/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_23/beta*
_output_shapes	
:?*
dtype0
?
batch_normalization_23/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_namebatch_normalization_23/gamma
?
0batch_normalization_23/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_23/gamma*
_output_shapes	
:?*
dtype0
?
&batch_normalization_22/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&batch_normalization_22/moving_variance
?
:batch_normalization_22/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_22/moving_variance*
_output_shapes	
:?*
dtype0
?
"batch_normalization_22/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"batch_normalization_22/moving_mean
?
6batch_normalization_22/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_22/moving_mean*
_output_shapes	
:?*
dtype0
?
batch_normalization_22/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namebatch_normalization_22/beta
?
/batch_normalization_22/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_22/beta*
_output_shapes	
:?*
dtype0
?
batch_normalization_22/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_namebatch_normalization_22/gamma
?
0batch_normalization_22/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_22/gamma*
_output_shapes	
:?*
dtype0
?
&batch_normalization_21/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_21/moving_variance
?
:batch_normalization_21/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_21/moving_variance*
_output_shapes
:@*
dtype0
?
"batch_normalization_21/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_21/moving_mean
?
6batch_normalization_21/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_21/moving_mean*
_output_shapes
:@*
dtype0
?
batch_normalization_21/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_21/beta
?
/batch_normalization_21/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_21/beta*
_output_shapes
:@*
dtype0
?
batch_normalization_21/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_21/gamma
?
0batch_normalization_21/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_21/gamma*
_output_shapes
:@*
dtype0
?
serving_default_input_5Placeholder*/
_output_shapes
:?????????*
dtype0*$
shape:?????????
?	
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_5"module_wrapper_85/conv2d_41/kernel module_wrapper_85/conv2d_41/bias"module_wrapper_86/conv2d_42/kernel module_wrapper_86/conv2d_42/biasbatch_normalization_21/gammabatch_normalization_21/beta"batch_normalization_21/moving_mean&batch_normalization_21/moving_variance"module_wrapper_88/conv2d_43/kernel module_wrapper_88/conv2d_43/bias"module_wrapper_89/conv2d_44/kernel module_wrapper_89/conv2d_44/biasbatch_normalization_22/gammabatch_normalization_22/beta"batch_normalization_22/moving_mean&batch_normalization_22/moving_variance"module_wrapper_91/conv2d_45/kernel module_wrapper_91/conv2d_45/biasbatch_normalization_23/gammabatch_normalization_23/beta"batch_normalization_23/moving_mean&batch_normalization_23/moving_variance!module_wrapper_94/dense_18/kernelmodule_wrapper_94/dense_18/bias!module_wrapper_95/dense_19/kernelmodule_wrapper_95/dense_19/bias*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_177220

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer-6
layer_with_weights-5
layer-7
	layer_with_weights-6
	layer-8

layer-9
layer_with_weights-7
layer-10
layer-11
layer_with_weights-8
layer-12
layer_with_weights-9
layer-13
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_module*
?
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses
%_module*
?
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses
,_module* 
?
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses
3axis
	4gamma
5beta
6moving_mean
7moving_variance*
?
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses
>_module*
?
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses
E_module*
?
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J__call__
*K&call_and_return_all_conditional_losses
L_module* 
?
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses
Saxis
	Tgamma
Ubeta
Vmoving_mean
Wmoving_variance*
?
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses
^_module*
?
_	variables
`trainable_variables
aregularization_losses
b	keras_api
c__call__
*d&call_and_return_all_conditional_losses
e_module* 
?
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
j__call__
*k&call_and_return_all_conditional_losses
laxis
	mgamma
nbeta
omoving_mean
pmoving_variance*
?
q	variables
rtrainable_variables
sregularization_losses
t	keras_api
u__call__
*v&call_and_return_all_conditional_losses
w_module* 
?
x	variables
ytrainable_variables
zregularization_losses
{	keras_api
|__call__
*}&call_and_return_all_conditional_losses
~_module*
?
	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_module*
?
?0
?1
?2
?3
44
55
66
77
?8
?9
?10
?11
T12
U13
V14
W15
?16
?17
m18
n19
o20
p21
?22
?23
?24
?25*
?
?0
?1
?2
?3
44
55
?6
?7
?8
?9
T10
U11
?12
?13
m14
n15
?16
?17
?18
?19*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
:
?trace_0
?trace_1
?trace_2
?trace_3* 
:
?trace_0
?trace_1
?trace_2
?trace_3* 
* 
?
?
_variables
?_iterations
?_learning_rate
?_index_dict
?
_momentums
?_velocities
?_update_step_xla*

?serving_default* 

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

?trace_0
?trace_1* 

?trace_0
?trace_1* 
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias*

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses*

?trace_0
?trace_1* 

?trace_0
?trace_1* 
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias*
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses* 

?trace_0
?trace_1* 

?trace_0
?trace_1* 
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
 
40
51
62
73*

40
51*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses*

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_21/gamma5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_21/beta4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_21/moving_mean;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_21/moving_variance?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses*

?trace_0
?trace_1* 

?trace_0
?trace_1* 
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias*

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses*

?trace_0
?trace_1* 

?trace_0
?trace_1* 
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias*
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses* 

?trace_0
?trace_1* 

?trace_0
?trace_1* 
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
 
T0
U1
V2
W3*

T0
U1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses*

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_22/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_22/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_22/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_22/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses*

?trace_0
?trace_1* 

?trace_0
?trace_1* 
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias*
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
_	variables
`trainable_variables
aregularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses* 

?trace_0
?trace_1* 

?trace_0
?trace_1* 
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
 
m0
n1
o2
p3*

m0
n1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
f	variables
gtrainable_variables
hregularization_losses
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses*

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_23/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_23/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_23/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_23/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
q	variables
rtrainable_variables
sregularization_losses
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses* 

?trace_0
?trace_1* 

?trace_0
?trace_1* 
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
x	variables
ytrainable_variables
zregularization_losses
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses*

?trace_0
?trace_1* 

?trace_0
?trace_1* 
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias*

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0
?trace_1* 

?trace_0
?trace_1* 
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias*
b\
VARIABLE_VALUE"module_wrapper_85/conv2d_41/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUE module_wrapper_85/conv2d_41/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE"module_wrapper_86/conv2d_42/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUE module_wrapper_86/conv2d_42/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE"module_wrapper_88/conv2d_43/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUE module_wrapper_88/conv2d_43/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE"module_wrapper_89/conv2d_44/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE module_wrapper_89/conv2d_44/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE"module_wrapper_91/conv2d_45/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE module_wrapper_91/conv2d_45/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE!module_wrapper_94/dense_18/kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEmodule_wrapper_94/dense_18/bias'variables/23/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE!module_wrapper_95/dense_19/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEmodule_wrapper_95/dense_19/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE*
.
60
71
V2
W3
o4
p5*
j
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
13*

?0
?1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
?
?0
?1
?2
?3
?4
?5
?6
?7
?8
?9
?10
?11
?12
?13
?14
?15
?16
?17
?18
?19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35
?36
?37
?38
?39
?40*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
?
?0
?1
?2
?3
?4
?5
?6
?7
?8
?9
?10
?11
?12
?13
?14
?15
?16
?17
?18
?19*
?
?0
?1
?2
?3
?4
?5
?6
?7
?8
?9
?10
?11
?12
?13
?14
?15
?16
?17
?18
?19*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

?0
?1*
* 

?0
?1*
?
?metrics
?	variables
?non_trainable_variables
?layer_metrics
?regularization_losses
?trainable_variables
?layers
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

?0
?1*
* 

?0
?1*
?
?metrics
?	variables
?non_trainable_variables
?layer_metrics
?regularization_losses
?trainable_variables
?layers
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
?
?metrics
?	variables
?non_trainable_variables
?layer_metrics
?regularization_losses
?trainable_variables
?layers
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 

60
71*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

?0
?1*
* 

?0
?1*
?
?metrics
?	variables
?non_trainable_variables
?layer_metrics
?regularization_losses
?trainable_variables
?layers
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

?0
?1*
* 

?0
?1*
?
?metrics
?	variables
?non_trainable_variables
?layer_metrics
?regularization_losses
?trainable_variables
?layers
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
?
?metrics
?	variables
?non_trainable_variables
?layer_metrics
?regularization_losses
?trainable_variables
?layers
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 

V0
W1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

?0
?1*
* 

?0
?1*
?
?metrics
?	variables
?non_trainable_variables
?layer_metrics
?regularization_losses
?trainable_variables
?layers
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
?
?metrics
?	variables
?non_trainable_variables
?layer_metrics
?regularization_losses
?trainable_variables
?layers
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 

o0
p1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
?
?metrics
?	variables
?non_trainable_variables
?layer_metrics
?regularization_losses
?trainable_variables
?layers
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

?0
?1*
* 

?0
?1*
?
?metrics
?	variables
?non_trainable_variables
?layer_metrics
?regularization_losses
?trainable_variables
?layers
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

?0
?1*
* 

?0
?1*
?
?metrics
?	variables
?non_trainable_variables
?layer_metrics
?regularization_losses
?trainable_variables
?layers
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
<
?	variables
?	keras_api

?total

?count*
M
?	variables
?	keras_api

?total

?count
?
_fn_kwargs*
tn
VARIABLE_VALUE)Adam/m/module_wrapper_85/conv2d_41/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE)Adam/v/module_wrapper_85/conv2d_41/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE'Adam/m/module_wrapper_85/conv2d_41/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE'Adam/v/module_wrapper_85/conv2d_41/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE)Adam/m/module_wrapper_86/conv2d_42/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE)Adam/v/module_wrapper_86/conv2d_42/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE'Adam/m/module_wrapper_86/conv2d_42/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE'Adam/v/module_wrapper_86/conv2d_42/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE#Adam/m/batch_normalization_21/gamma1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE#Adam/v/batch_normalization_21/gamma2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/batch_normalization_21/beta2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/batch_normalization_21/beta2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE)Adam/m/module_wrapper_88/conv2d_43/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE)Adam/v/module_wrapper_88/conv2d_43/kernel2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE'Adam/m/module_wrapper_88/conv2d_43/bias2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE'Adam/v/module_wrapper_88/conv2d_43/bias2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE)Adam/m/module_wrapper_89/conv2d_44/kernel2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE)Adam/v/module_wrapper_89/conv2d_44/kernel2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE'Adam/m/module_wrapper_89/conv2d_44/bias2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE'Adam/v/module_wrapper_89/conv2d_44/bias2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE#Adam/m/batch_normalization_22/gamma2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE#Adam/v/batch_normalization_22/gamma2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/batch_normalization_22/beta2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/batch_normalization_22/beta2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE)Adam/m/module_wrapper_91/conv2d_45/kernel2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE)Adam/v/module_wrapper_91/conv2d_45/kernel2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE'Adam/m/module_wrapper_91/conv2d_45/bias2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE'Adam/v/module_wrapper_91/conv2d_45/bias2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE#Adam/m/batch_normalization_23/gamma2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE#Adam/v/batch_normalization_23/gamma2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/batch_normalization_23/beta2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/batch_normalization_23/beta2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE(Adam/m/module_wrapper_94/dense_18/kernel2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE(Adam/v/module_wrapper_94/dense_18/kernel2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE&Adam/m/module_wrapper_94/dense_18/bias2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE&Adam/v/module_wrapper_94/dense_18/bias2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE(Adam/m/module_wrapper_95/dense_19/kernel2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE(Adam/v/module_wrapper_95/dense_19/kernel2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE&Adam/m/module_wrapper_95/dense_19/bias2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE&Adam/v/module_wrapper_95/dense_19/bias2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

?0
?1*

?	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?!
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename0batch_normalization_21/gamma/Read/ReadVariableOp/batch_normalization_21/beta/Read/ReadVariableOp6batch_normalization_21/moving_mean/Read/ReadVariableOp:batch_normalization_21/moving_variance/Read/ReadVariableOp0batch_normalization_22/gamma/Read/ReadVariableOp/batch_normalization_22/beta/Read/ReadVariableOp6batch_normalization_22/moving_mean/Read/ReadVariableOp:batch_normalization_22/moving_variance/Read/ReadVariableOp0batch_normalization_23/gamma/Read/ReadVariableOp/batch_normalization_23/beta/Read/ReadVariableOp6batch_normalization_23/moving_mean/Read/ReadVariableOp:batch_normalization_23/moving_variance/Read/ReadVariableOp6module_wrapper_85/conv2d_41/kernel/Read/ReadVariableOp4module_wrapper_85/conv2d_41/bias/Read/ReadVariableOp6module_wrapper_86/conv2d_42/kernel/Read/ReadVariableOp4module_wrapper_86/conv2d_42/bias/Read/ReadVariableOp6module_wrapper_88/conv2d_43/kernel/Read/ReadVariableOp4module_wrapper_88/conv2d_43/bias/Read/ReadVariableOp6module_wrapper_89/conv2d_44/kernel/Read/ReadVariableOp4module_wrapper_89/conv2d_44/bias/Read/ReadVariableOp6module_wrapper_91/conv2d_45/kernel/Read/ReadVariableOp4module_wrapper_91/conv2d_45/bias/Read/ReadVariableOp5module_wrapper_94/dense_18/kernel/Read/ReadVariableOp3module_wrapper_94/dense_18/bias/Read/ReadVariableOp5module_wrapper_95/dense_19/kernel/Read/ReadVariableOp3module_wrapper_95/dense_19/bias/Read/ReadVariableOpiteration/Read/ReadVariableOp!learning_rate/Read/ReadVariableOp=Adam/m/module_wrapper_85/conv2d_41/kernel/Read/ReadVariableOp=Adam/v/module_wrapper_85/conv2d_41/kernel/Read/ReadVariableOp;Adam/m/module_wrapper_85/conv2d_41/bias/Read/ReadVariableOp;Adam/v/module_wrapper_85/conv2d_41/bias/Read/ReadVariableOp=Adam/m/module_wrapper_86/conv2d_42/kernel/Read/ReadVariableOp=Adam/v/module_wrapper_86/conv2d_42/kernel/Read/ReadVariableOp;Adam/m/module_wrapper_86/conv2d_42/bias/Read/ReadVariableOp;Adam/v/module_wrapper_86/conv2d_42/bias/Read/ReadVariableOp7Adam/m/batch_normalization_21/gamma/Read/ReadVariableOp7Adam/v/batch_normalization_21/gamma/Read/ReadVariableOp6Adam/m/batch_normalization_21/beta/Read/ReadVariableOp6Adam/v/batch_normalization_21/beta/Read/ReadVariableOp=Adam/m/module_wrapper_88/conv2d_43/kernel/Read/ReadVariableOp=Adam/v/module_wrapper_88/conv2d_43/kernel/Read/ReadVariableOp;Adam/m/module_wrapper_88/conv2d_43/bias/Read/ReadVariableOp;Adam/v/module_wrapper_88/conv2d_43/bias/Read/ReadVariableOp=Adam/m/module_wrapper_89/conv2d_44/kernel/Read/ReadVariableOp=Adam/v/module_wrapper_89/conv2d_44/kernel/Read/ReadVariableOp;Adam/m/module_wrapper_89/conv2d_44/bias/Read/ReadVariableOp;Adam/v/module_wrapper_89/conv2d_44/bias/Read/ReadVariableOp7Adam/m/batch_normalization_22/gamma/Read/ReadVariableOp7Adam/v/batch_normalization_22/gamma/Read/ReadVariableOp6Adam/m/batch_normalization_22/beta/Read/ReadVariableOp6Adam/v/batch_normalization_22/beta/Read/ReadVariableOp=Adam/m/module_wrapper_91/conv2d_45/kernel/Read/ReadVariableOp=Adam/v/module_wrapper_91/conv2d_45/kernel/Read/ReadVariableOp;Adam/m/module_wrapper_91/conv2d_45/bias/Read/ReadVariableOp;Adam/v/module_wrapper_91/conv2d_45/bias/Read/ReadVariableOp7Adam/m/batch_normalization_23/gamma/Read/ReadVariableOp7Adam/v/batch_normalization_23/gamma/Read/ReadVariableOp6Adam/m/batch_normalization_23/beta/Read/ReadVariableOp6Adam/v/batch_normalization_23/beta/Read/ReadVariableOp<Adam/m/module_wrapper_94/dense_18/kernel/Read/ReadVariableOp<Adam/v/module_wrapper_94/dense_18/kernel/Read/ReadVariableOp:Adam/m/module_wrapper_94/dense_18/bias/Read/ReadVariableOp:Adam/v/module_wrapper_94/dense_18/bias/Read/ReadVariableOp<Adam/m/module_wrapper_95/dense_19/kernel/Read/ReadVariableOp<Adam/v/module_wrapper_95/dense_19/kernel/Read/ReadVariableOp:Adam/m/module_wrapper_95/dense_19/bias/Read/ReadVariableOp:Adam/v/module_wrapper_95/dense_19/bias/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*U
TinN
L2J	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__traced_save_178358
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamebatch_normalization_21/gammabatch_normalization_21/beta"batch_normalization_21/moving_mean&batch_normalization_21/moving_variancebatch_normalization_22/gammabatch_normalization_22/beta"batch_normalization_22/moving_mean&batch_normalization_22/moving_variancebatch_normalization_23/gammabatch_normalization_23/beta"batch_normalization_23/moving_mean&batch_normalization_23/moving_variance"module_wrapper_85/conv2d_41/kernel module_wrapper_85/conv2d_41/bias"module_wrapper_86/conv2d_42/kernel module_wrapper_86/conv2d_42/bias"module_wrapper_88/conv2d_43/kernel module_wrapper_88/conv2d_43/bias"module_wrapper_89/conv2d_44/kernel module_wrapper_89/conv2d_44/bias"module_wrapper_91/conv2d_45/kernel module_wrapper_91/conv2d_45/bias!module_wrapper_94/dense_18/kernelmodule_wrapper_94/dense_18/bias!module_wrapper_95/dense_19/kernelmodule_wrapper_95/dense_19/bias	iterationlearning_rate)Adam/m/module_wrapper_85/conv2d_41/kernel)Adam/v/module_wrapper_85/conv2d_41/kernel'Adam/m/module_wrapper_85/conv2d_41/bias'Adam/v/module_wrapper_85/conv2d_41/bias)Adam/m/module_wrapper_86/conv2d_42/kernel)Adam/v/module_wrapper_86/conv2d_42/kernel'Adam/m/module_wrapper_86/conv2d_42/bias'Adam/v/module_wrapper_86/conv2d_42/bias#Adam/m/batch_normalization_21/gamma#Adam/v/batch_normalization_21/gamma"Adam/m/batch_normalization_21/beta"Adam/v/batch_normalization_21/beta)Adam/m/module_wrapper_88/conv2d_43/kernel)Adam/v/module_wrapper_88/conv2d_43/kernel'Adam/m/module_wrapper_88/conv2d_43/bias'Adam/v/module_wrapper_88/conv2d_43/bias)Adam/m/module_wrapper_89/conv2d_44/kernel)Adam/v/module_wrapper_89/conv2d_44/kernel'Adam/m/module_wrapper_89/conv2d_44/bias'Adam/v/module_wrapper_89/conv2d_44/bias#Adam/m/batch_normalization_22/gamma#Adam/v/batch_normalization_22/gamma"Adam/m/batch_normalization_22/beta"Adam/v/batch_normalization_22/beta)Adam/m/module_wrapper_91/conv2d_45/kernel)Adam/v/module_wrapper_91/conv2d_45/kernel'Adam/m/module_wrapper_91/conv2d_45/bias'Adam/v/module_wrapper_91/conv2d_45/bias#Adam/m/batch_normalization_23/gamma#Adam/v/batch_normalization_23/gamma"Adam/m/batch_normalization_23/beta"Adam/v/batch_normalization_23/beta(Adam/m/module_wrapper_94/dense_18/kernel(Adam/v/module_wrapper_94/dense_18/kernel&Adam/m/module_wrapper_94/dense_18/bias&Adam/v/module_wrapper_94/dense_18/bias(Adam/m/module_wrapper_95/dense_19/kernel(Adam/v/module_wrapper_95/dense_19/kernel&Adam/m/module_wrapper_95/dense_19/bias&Adam/v/module_wrapper_95/dense_19/biastotal_1count_1totalcount*T
TinM
K2I*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__traced_restore_178584??
?
?
2__inference_module_wrapper_88_layer_call_fn_177727

args_0"
unknown:@?
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????

?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_88_layer_call_and_return_conditional_losses_176695x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????

?`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameargs_0
?
?
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_177883

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_177865

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
N
2__inference_module_wrapper_87_layer_call_fn_177620

args_0
identity?
PartitionedCallPartitionedCallargs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_87_layer_call_and_return_conditional_losses_176311h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameargs_0
?
?
2__inference_module_wrapper_85_layer_call_fn_177552

args_0!
unknown:@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_85_layer_call_and_return_conditional_losses_176771w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameargs_0
?
N
2__inference_module_wrapper_92_layer_call_fn_177933

args_0
identity?
PartitionedCallPartitionedCallargs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_92_layer_call_and_return_conditional_losses_176593i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameargs_0
?
N
2__inference_module_wrapper_92_layer_call_fn_177928

args_0
identity?
PartitionedCallPartitionedCallargs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_92_layer_call_and_return_conditional_losses_176394i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameargs_0
?
i
M__inference_module_wrapper_93_layer_call_and_return_conditional_losses_178039

args_0
identity`
flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   q
flatten_9/ReshapeReshapeargs_0flatten_9/Const:output:0*
T0*(
_output_shapes
:??????????c
IdentityIdentityflatten_9/Reshape:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameargs_0
?
h
L__inference_max_pooling2d_27_layer_call_and_return_conditional_losses_177815

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
M__inference_module_wrapper_91_layer_call_and_return_conditional_losses_176383

args_0D
(conv2d_45_conv2d_readvariableop_resource:??8
)conv2d_45_biasadd_readvariableop_resource:	?
identity?? conv2d_45/BiasAdd/ReadVariableOp?conv2d_45/Conv2D/ReadVariableOp?
conv2d_45/Conv2D/ReadVariableOpReadVariableOp(conv2d_45_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
conv2d_45/Conv2DConv2Dargs_0'conv2d_45/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
?
 conv2d_45/BiasAdd/ReadVariableOpReadVariableOp)conv2d_45_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
conv2d_45/BiasAddBiasAddconv2d_45/Conv2D:output:0(conv2d_45/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????m
conv2d_45/ReluReluconv2d_45/BiasAdd:output:0*
T0*0
_output_shapes
:??????????t
IdentityIdentityconv2d_45/Relu:activations:0^NoOp*
T0*0
_output_shapes
:???????????
NoOpNoOp!^conv2d_45/BiasAdd/ReadVariableOp ^conv2d_45/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 2D
 conv2d_45/BiasAdd/ReadVariableOp conv2d_45/BiasAdd/ReadVariableOp2B
conv2d_45/Conv2D/ReadVariableOpconv2d_45/Conv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameargs_0
?
i
M__inference_module_wrapper_92_layer_call_and_return_conditional_losses_176394

args_0
identity?
max_pooling2d_28/MaxPoolMaxPoolargs_0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
r
IdentityIdentity!max_pooling2d_28/MaxPool:output:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameargs_0
?
?
M__inference_module_wrapper_85_layer_call_and_return_conditional_losses_177563

args_0B
(conv2d_41_conv2d_readvariableop_resource:@7
)conv2d_41_biasadd_readvariableop_resource:@
identity?? conv2d_41/BiasAdd/ReadVariableOp?conv2d_41/Conv2D/ReadVariableOp?
conv2d_41/Conv2D/ReadVariableOpReadVariableOp(conv2d_41_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
conv2d_41/Conv2DConv2Dargs_0'conv2d_41/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
?
 conv2d_41/BiasAdd/ReadVariableOpReadVariableOp)conv2d_41_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_41/BiasAddBiasAddconv2d_41/Conv2D:output:0(conv2d_41/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@l
conv2d_41/ReluReluconv2d_41/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@s
IdentityIdentityconv2d_41/Relu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@?
NoOpNoOp!^conv2d_41/BiasAdd/ReadVariableOp ^conv2d_41/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 2D
 conv2d_41/BiasAdd/ReadVariableOp conv2d_41/BiasAdd/ReadVariableOp2B
conv2d_41/Conv2D/ReadVariableOpconv2d_41/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameargs_0
?
?
M__inference_module_wrapper_86_layer_call_and_return_conditional_losses_176300

args_0B
(conv2d_42_conv2d_readvariableop_resource:@@7
)conv2d_42_biasadd_readvariableop_resource:@
identity?? conv2d_42/BiasAdd/ReadVariableOp?conv2d_42/Conv2D/ReadVariableOp?
conv2d_42/Conv2D/ReadVariableOpReadVariableOp(conv2d_42_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
conv2d_42/Conv2DConv2Dargs_0'conv2d_42/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
?
 conv2d_42/BiasAdd/ReadVariableOpReadVariableOp)conv2d_42_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_42/BiasAddBiasAddconv2d_42/Conv2D:output:0(conv2d_42/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@l
conv2d_42/ReluReluconv2d_42/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@s
IdentityIdentityconv2d_42/Relu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@?
NoOpNoOp!^conv2d_42/BiasAdd/ReadVariableOp ^conv2d_42/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 2D
 conv2d_42/BiasAdd/ReadVariableOp conv2d_42/BiasAdd/ReadVariableOp2B
conv2d_42/Conv2D/ReadVariableOpconv2d_42/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameargs_0
?
?
-__inference_sequential_9_layer_call_fn_177334

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@
	unknown_6:@$
	unknown_7:@?
	unknown_8:	?%
	unknown_9:??

unknown_10:	?

unknown_11:	?

unknown_12:	?

unknown_13:	?

unknown_14:	?&

unknown_15:??

unknown_16:	?

unknown_17:	?

unknown_18:	?

unknown_19:	?

unknown_20:	?

unknown_21:
??

unknown_22:	?

unknown_23:	?

unknown_24:
identity??StatefulPartitionedCall?
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
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_9_layer_call_and_return_conditional_losses_176907o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
M__inference_module_wrapper_94_layer_call_and_return_conditional_losses_178068

args_0;
'dense_18_matmul_readvariableop_resource:
??7
(dense_18_biasadd_readvariableop_resource:	?
identity??dense_18/BiasAdd/ReadVariableOp?dense_18/MatMul/ReadVariableOp?
dense_18/MatMul/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0|
dense_18/MatMulMatMulargs_0&dense_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_18/BiasAdd/ReadVariableOpReadVariableOp(dense_18_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_18/BiasAddBiasAdddense_18/MatMul:product:0'dense_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????c
dense_18/ReluReludense_18/BiasAdd:output:0*
T0*(
_output_shapes
:??????????k
IdentityIdentitydense_18/Relu:activations:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 2B
dense_18/BiasAdd/ReadVariableOpdense_18/BiasAdd/ReadVariableOp2@
dense_18/MatMul/ReadVariableOpdense_18/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameargs_0
?
?
M__inference_module_wrapper_85_layer_call_and_return_conditional_losses_176771

args_0B
(conv2d_41_conv2d_readvariableop_resource:@7
)conv2d_41_biasadd_readvariableop_resource:@
identity?? conv2d_41/BiasAdd/ReadVariableOp?conv2d_41/Conv2D/ReadVariableOp?
conv2d_41/Conv2D/ReadVariableOpReadVariableOp(conv2d_41_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
conv2d_41/Conv2DConv2Dargs_0'conv2d_41/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
?
 conv2d_41/BiasAdd/ReadVariableOpReadVariableOp)conv2d_41_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_41/BiasAddBiasAddconv2d_41/Conv2D:output:0(conv2d_41/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@l
conv2d_41/ReluReluconv2d_41/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@s
IdentityIdentityconv2d_41/Relu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@?
NoOpNoOp!^conv2d_41/BiasAdd/ReadVariableOp ^conv2d_41/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 2D
 conv2d_41/BiasAdd/ReadVariableOp conv2d_41/BiasAdd/ReadVariableOp2B
conv2d_41/Conv2D/ReadVariableOpconv2d_41/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameargs_0
?
?
-__inference_sequential_9_layer_call_fn_177277

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@
	unknown_6:@$
	unknown_7:@?
	unknown_8:	?%
	unknown_9:??

unknown_10:	?

unknown_11:	?

unknown_12:	?

unknown_13:	?

unknown_14:	?&

unknown_15:??

unknown_16:	?

unknown_17:	?

unknown_18:	?

unknown_19:	?

unknown_20:	?

unknown_21:
??

unknown_22:	?

unknown_23:	?

unknown_24:
identity??StatefulPartitionedCall?
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
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_9_layer_call_and_return_conditional_losses_176448o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
i
M__inference_module_wrapper_90_layer_call_and_return_conditional_losses_177809

args_0
identity?
max_pooling2d_27/MaxPoolMaxPoolargs_0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
r
IdentityIdentity!max_pooling2d_27/MaxPool:output:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameargs_0
?
?
M__inference_module_wrapper_95_layer_call_and_return_conditional_losses_178108

args_0:
'dense_19_matmul_readvariableop_resource:	?6
(dense_19_biasadd_readvariableop_resource:
identity??dense_19/BiasAdd/ReadVariableOp?dense_19/MatMul/ReadVariableOp?
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0{
dense_19/MatMulMatMulargs_0&dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_19/BiasAdd/ReadVariableOpReadVariableOp(dense_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_19/BiasAddBiasAdddense_19/MatMul:product:0'dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????h
dense_19/SoftmaxSoftmaxdense_19/BiasAdd:output:0*
T0*'
_output_shapes
:?????????i
IdentityIdentitydense_19/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 2B
dense_19/BiasAdd/ReadVariableOpdense_19/BiasAdd/ReadVariableOp2@
dense_19/MatMul/ReadVariableOpdense_19/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameargs_0
?
N
2__inference_module_wrapper_93_layer_call_fn_178022

args_0
identity?
PartitionedCallPartitionedCallargs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_93_layer_call_and_return_conditional_losses_176411a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameargs_0
?
?
2__inference_module_wrapper_89_layer_call_fn_177767

args_0#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_89_layer_call_and_return_conditional_losses_176665x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????

?: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????

?
 
_user_specified_nameargs_0
?
?
M__inference_module_wrapper_86_layer_call_and_return_conditional_losses_177615

args_0B
(conv2d_42_conv2d_readvariableop_resource:@@7
)conv2d_42_biasadd_readvariableop_resource:@
identity?? conv2d_42/BiasAdd/ReadVariableOp?conv2d_42/Conv2D/ReadVariableOp?
conv2d_42/Conv2D/ReadVariableOpReadVariableOp(conv2d_42_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
conv2d_42/Conv2DConv2Dargs_0'conv2d_42/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
?
 conv2d_42/BiasAdd/ReadVariableOpReadVariableOp)conv2d_42_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_42/BiasAddBiasAddconv2d_42/Conv2D:output:0(conv2d_42/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@l
conv2d_42/ReluReluconv2d_42/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@s
IdentityIdentityconv2d_42/Relu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@?
NoOpNoOp!^conv2d_42/BiasAdd/ReadVariableOp ^conv2d_42/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 2D
 conv2d_42/BiasAdd/ReadVariableOp conv2d_42/BiasAdd/ReadVariableOp2B
conv2d_42/Conv2D/ReadVariableOpconv2d_42/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameargs_0
?
?
M__inference_module_wrapper_86_layer_call_and_return_conditional_losses_176741

args_0B
(conv2d_42_conv2d_readvariableop_resource:@@7
)conv2d_42_biasadd_readvariableop_resource:@
identity?? conv2d_42/BiasAdd/ReadVariableOp?conv2d_42/Conv2D/ReadVariableOp?
conv2d_42/Conv2D/ReadVariableOpReadVariableOp(conv2d_42_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
conv2d_42/Conv2DConv2Dargs_0'conv2d_42/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
?
 conv2d_42/BiasAdd/ReadVariableOpReadVariableOp)conv2d_42_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_42/BiasAddBiasAddconv2d_42/Conv2D:output:0(conv2d_42/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@l
conv2d_42/ReluReluconv2d_42/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@s
IdentityIdentityconv2d_42/Relu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@?
NoOpNoOp!^conv2d_42/BiasAdd/ReadVariableOp ^conv2d_42/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 2D
 conv2d_42/BiasAdd/ReadVariableOp conv2d_42/BiasAdd/ReadVariableOp2B
conv2d_42/Conv2D/ReadVariableOpconv2d_42/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameargs_0
?
?
M__inference_module_wrapper_88_layer_call_and_return_conditional_losses_176695

args_0C
(conv2d_43_conv2d_readvariableop_resource:@?8
)conv2d_43_biasadd_readvariableop_resource:	?
identity?? conv2d_43/BiasAdd/ReadVariableOp?conv2d_43/Conv2D/ReadVariableOp?
conv2d_43/Conv2D/ReadVariableOpReadVariableOp(conv2d_43_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
conv2d_43/Conv2DConv2Dargs_0'conv2d_43/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????

?*
paddingVALID*
strides
?
 conv2d_43/BiasAdd/ReadVariableOpReadVariableOp)conv2d_43_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
conv2d_43/BiasAddBiasAddconv2d_43/Conv2D:output:0(conv2d_43/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????

?m
conv2d_43/ReluReluconv2d_43/BiasAdd:output:0*
T0*0
_output_shapes
:?????????

?t
IdentityIdentityconv2d_43/Relu:activations:0^NoOp*
T0*0
_output_shapes
:?????????

??
NoOpNoOp!^conv2d_43/BiasAdd/ReadVariableOp ^conv2d_43/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 2D
 conv2d_43/BiasAdd/ReadVariableOp conv2d_43/BiasAdd/ReadVariableOp2B
conv2d_43/Conv2D/ReadVariableOpconv2d_43/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameargs_0
??
?
H__inference_sequential_9_layer_call_and_return_conditional_losses_177434

inputsT
:module_wrapper_85_conv2d_41_conv2d_readvariableop_resource:@I
;module_wrapper_85_conv2d_41_biasadd_readvariableop_resource:@T
:module_wrapper_86_conv2d_42_conv2d_readvariableop_resource:@@I
;module_wrapper_86_conv2d_42_biasadd_readvariableop_resource:@<
.batch_normalization_21_readvariableop_resource:@>
0batch_normalization_21_readvariableop_1_resource:@M
?batch_normalization_21_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_21_fusedbatchnormv3_readvariableop_1_resource:@U
:module_wrapper_88_conv2d_43_conv2d_readvariableop_resource:@?J
;module_wrapper_88_conv2d_43_biasadd_readvariableop_resource:	?V
:module_wrapper_89_conv2d_44_conv2d_readvariableop_resource:??J
;module_wrapper_89_conv2d_44_biasadd_readvariableop_resource:	?=
.batch_normalization_22_readvariableop_resource:	??
0batch_normalization_22_readvariableop_1_resource:	?N
?batch_normalization_22_fusedbatchnormv3_readvariableop_resource:	?P
Abatch_normalization_22_fusedbatchnormv3_readvariableop_1_resource:	?V
:module_wrapper_91_conv2d_45_conv2d_readvariableop_resource:??J
;module_wrapper_91_conv2d_45_biasadd_readvariableop_resource:	?=
.batch_normalization_23_readvariableop_resource:	??
0batch_normalization_23_readvariableop_1_resource:	?N
?batch_normalization_23_fusedbatchnormv3_readvariableop_resource:	?P
Abatch_normalization_23_fusedbatchnormv3_readvariableop_1_resource:	?M
9module_wrapper_94_dense_18_matmul_readvariableop_resource:
??I
:module_wrapper_94_dense_18_biasadd_readvariableop_resource:	?L
9module_wrapper_95_dense_19_matmul_readvariableop_resource:	?H
:module_wrapper_95_dense_19_biasadd_readvariableop_resource:
identity??6batch_normalization_21/FusedBatchNormV3/ReadVariableOp?8batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_21/ReadVariableOp?'batch_normalization_21/ReadVariableOp_1?6batch_normalization_22/FusedBatchNormV3/ReadVariableOp?8batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_22/ReadVariableOp?'batch_normalization_22/ReadVariableOp_1?6batch_normalization_23/FusedBatchNormV3/ReadVariableOp?8batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_23/ReadVariableOp?'batch_normalization_23/ReadVariableOp_1?2module_wrapper_85/conv2d_41/BiasAdd/ReadVariableOp?1module_wrapper_85/conv2d_41/Conv2D/ReadVariableOp?2module_wrapper_86/conv2d_42/BiasAdd/ReadVariableOp?1module_wrapper_86/conv2d_42/Conv2D/ReadVariableOp?2module_wrapper_88/conv2d_43/BiasAdd/ReadVariableOp?1module_wrapper_88/conv2d_43/Conv2D/ReadVariableOp?2module_wrapper_89/conv2d_44/BiasAdd/ReadVariableOp?1module_wrapper_89/conv2d_44/Conv2D/ReadVariableOp?2module_wrapper_91/conv2d_45/BiasAdd/ReadVariableOp?1module_wrapper_91/conv2d_45/Conv2D/ReadVariableOp?1module_wrapper_94/dense_18/BiasAdd/ReadVariableOp?0module_wrapper_94/dense_18/MatMul/ReadVariableOp?1module_wrapper_95/dense_19/BiasAdd/ReadVariableOp?0module_wrapper_95/dense_19/MatMul/ReadVariableOp?
1module_wrapper_85/conv2d_41/Conv2D/ReadVariableOpReadVariableOp:module_wrapper_85_conv2d_41_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
"module_wrapper_85/conv2d_41/Conv2DConv2Dinputs9module_wrapper_85/conv2d_41/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
?
2module_wrapper_85/conv2d_41/BiasAdd/ReadVariableOpReadVariableOp;module_wrapper_85_conv2d_41_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
#module_wrapper_85/conv2d_41/BiasAddBiasAdd+module_wrapper_85/conv2d_41/Conv2D:output:0:module_wrapper_85/conv2d_41/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@?
 module_wrapper_85/conv2d_41/ReluRelu,module_wrapper_85/conv2d_41/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@?
1module_wrapper_86/conv2d_42/Conv2D/ReadVariableOpReadVariableOp:module_wrapper_86_conv2d_42_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
"module_wrapper_86/conv2d_42/Conv2DConv2D.module_wrapper_85/conv2d_41/Relu:activations:09module_wrapper_86/conv2d_42/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
?
2module_wrapper_86/conv2d_42/BiasAdd/ReadVariableOpReadVariableOp;module_wrapper_86_conv2d_42_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
#module_wrapper_86/conv2d_42/BiasAddBiasAdd+module_wrapper_86/conv2d_42/Conv2D:output:0:module_wrapper_86/conv2d_42/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@?
 module_wrapper_86/conv2d_42/ReluRelu,module_wrapper_86/conv2d_42/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@?
*module_wrapper_87/max_pooling2d_26/MaxPoolMaxPool.module_wrapper_86/conv2d_42/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
?
%batch_normalization_21/ReadVariableOpReadVariableOp.batch_normalization_21_readvariableop_resource*
_output_shapes
:@*
dtype0?
'batch_normalization_21/ReadVariableOp_1ReadVariableOp0batch_normalization_21_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
6batch_normalization_21/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_21_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
8batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_21_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
'batch_normalization_21/FusedBatchNormV3FusedBatchNormV33module_wrapper_87/max_pooling2d_26/MaxPool:output:0-batch_normalization_21/ReadVariableOp:value:0/batch_normalization_21/ReadVariableOp_1:value:0>batch_normalization_21/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
is_training( ?
1module_wrapper_88/conv2d_43/Conv2D/ReadVariableOpReadVariableOp:module_wrapper_88_conv2d_43_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
"module_wrapper_88/conv2d_43/Conv2DConv2D+batch_normalization_21/FusedBatchNormV3:y:09module_wrapper_88/conv2d_43/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????

?*
paddingVALID*
strides
?
2module_wrapper_88/conv2d_43/BiasAdd/ReadVariableOpReadVariableOp;module_wrapper_88_conv2d_43_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
#module_wrapper_88/conv2d_43/BiasAddBiasAdd+module_wrapper_88/conv2d_43/Conv2D:output:0:module_wrapper_88/conv2d_43/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????

??
 module_wrapper_88/conv2d_43/ReluRelu,module_wrapper_88/conv2d_43/BiasAdd:output:0*
T0*0
_output_shapes
:?????????

??
1module_wrapper_89/conv2d_44/Conv2D/ReadVariableOpReadVariableOp:module_wrapper_89_conv2d_44_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
"module_wrapper_89/conv2d_44/Conv2DConv2D.module_wrapper_88/conv2d_43/Relu:activations:09module_wrapper_89/conv2d_44/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
?
2module_wrapper_89/conv2d_44/BiasAdd/ReadVariableOpReadVariableOp;module_wrapper_89_conv2d_44_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
#module_wrapper_89/conv2d_44/BiasAddBiasAdd+module_wrapper_89/conv2d_44/Conv2D:output:0:module_wrapper_89/conv2d_44/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:???????????
 module_wrapper_89/conv2d_44/ReluRelu,module_wrapper_89/conv2d_44/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
*module_wrapper_90/max_pooling2d_27/MaxPoolMaxPool.module_wrapper_89/conv2d_44/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
?
%batch_normalization_22/ReadVariableOpReadVariableOp.batch_normalization_22_readvariableop_resource*
_output_shapes	
:?*
dtype0?
'batch_normalization_22/ReadVariableOp_1ReadVariableOp0batch_normalization_22_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
6batch_normalization_22/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_22_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
8batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_22_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
'batch_normalization_22/FusedBatchNormV3FusedBatchNormV33module_wrapper_90/max_pooling2d_27/MaxPool:output:0-batch_normalization_22/ReadVariableOp:value:0/batch_normalization_22/ReadVariableOp_1:value:0>batch_normalization_22/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( ?
1module_wrapper_91/conv2d_45/Conv2D/ReadVariableOpReadVariableOp:module_wrapper_91_conv2d_45_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
"module_wrapper_91/conv2d_45/Conv2DConv2D+batch_normalization_22/FusedBatchNormV3:y:09module_wrapper_91/conv2d_45/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
?
2module_wrapper_91/conv2d_45/BiasAdd/ReadVariableOpReadVariableOp;module_wrapper_91_conv2d_45_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
#module_wrapper_91/conv2d_45/BiasAddBiasAdd+module_wrapper_91/conv2d_45/Conv2D:output:0:module_wrapper_91/conv2d_45/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:???????????
 module_wrapper_91/conv2d_45/ReluRelu,module_wrapper_91/conv2d_45/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
*module_wrapper_92/max_pooling2d_28/MaxPoolMaxPool.module_wrapper_91/conv2d_45/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
?
%batch_normalization_23/ReadVariableOpReadVariableOp.batch_normalization_23_readvariableop_resource*
_output_shapes	
:?*
dtype0?
'batch_normalization_23/ReadVariableOp_1ReadVariableOp0batch_normalization_23_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
6batch_normalization_23/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_23_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
8batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_23_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
'batch_normalization_23/FusedBatchNormV3FusedBatchNormV33module_wrapper_92/max_pooling2d_28/MaxPool:output:0-batch_normalization_23/ReadVariableOp:value:0/batch_normalization_23/ReadVariableOp_1:value:0>batch_normalization_23/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( r
!module_wrapper_93/flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ?
#module_wrapper_93/flatten_9/ReshapeReshape+batch_normalization_23/FusedBatchNormV3:y:0*module_wrapper_93/flatten_9/Const:output:0*
T0*(
_output_shapes
:???????????
0module_wrapper_94/dense_18/MatMul/ReadVariableOpReadVariableOp9module_wrapper_94_dense_18_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
!module_wrapper_94/dense_18/MatMulMatMul,module_wrapper_93/flatten_9/Reshape:output:08module_wrapper_94/dense_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
1module_wrapper_94/dense_18/BiasAdd/ReadVariableOpReadVariableOp:module_wrapper_94_dense_18_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
"module_wrapper_94/dense_18/BiasAddBiasAdd+module_wrapper_94/dense_18/MatMul:product:09module_wrapper_94/dense_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
module_wrapper_94/dense_18/ReluRelu+module_wrapper_94/dense_18/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
0module_wrapper_95/dense_19/MatMul/ReadVariableOpReadVariableOp9module_wrapper_95_dense_19_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
!module_wrapper_95/dense_19/MatMulMatMul-module_wrapper_94/dense_18/Relu:activations:08module_wrapper_95/dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
1module_wrapper_95/dense_19/BiasAdd/ReadVariableOpReadVariableOp:module_wrapper_95_dense_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
"module_wrapper_95/dense_19/BiasAddBiasAdd+module_wrapper_95/dense_19/MatMul:product:09module_wrapper_95/dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
"module_wrapper_95/dense_19/SoftmaxSoftmax+module_wrapper_95/dense_19/BiasAdd:output:0*
T0*'
_output_shapes
:?????????{
IdentityIdentity,module_wrapper_95/dense_19/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????

NoOpNoOp7^batch_normalization_21/FusedBatchNormV3/ReadVariableOp9^batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_21/ReadVariableOp(^batch_normalization_21/ReadVariableOp_17^batch_normalization_22/FusedBatchNormV3/ReadVariableOp9^batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_22/ReadVariableOp(^batch_normalization_22/ReadVariableOp_17^batch_normalization_23/FusedBatchNormV3/ReadVariableOp9^batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_23/ReadVariableOp(^batch_normalization_23/ReadVariableOp_13^module_wrapper_85/conv2d_41/BiasAdd/ReadVariableOp2^module_wrapper_85/conv2d_41/Conv2D/ReadVariableOp3^module_wrapper_86/conv2d_42/BiasAdd/ReadVariableOp2^module_wrapper_86/conv2d_42/Conv2D/ReadVariableOp3^module_wrapper_88/conv2d_43/BiasAdd/ReadVariableOp2^module_wrapper_88/conv2d_43/Conv2D/ReadVariableOp3^module_wrapper_89/conv2d_44/BiasAdd/ReadVariableOp2^module_wrapper_89/conv2d_44/Conv2D/ReadVariableOp3^module_wrapper_91/conv2d_45/BiasAdd/ReadVariableOp2^module_wrapper_91/conv2d_45/Conv2D/ReadVariableOp2^module_wrapper_94/dense_18/BiasAdd/ReadVariableOp1^module_wrapper_94/dense_18/MatMul/ReadVariableOp2^module_wrapper_95/dense_19/BiasAdd/ReadVariableOp1^module_wrapper_95/dense_19/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : 2p
6batch_normalization_21/FusedBatchNormV3/ReadVariableOp6batch_normalization_21/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_21/FusedBatchNormV3/ReadVariableOp_18batch_normalization_21/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_21/ReadVariableOp%batch_normalization_21/ReadVariableOp2R
'batch_normalization_21/ReadVariableOp_1'batch_normalization_21/ReadVariableOp_12p
6batch_normalization_22/FusedBatchNormV3/ReadVariableOp6batch_normalization_22/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_22/FusedBatchNormV3/ReadVariableOp_18batch_normalization_22/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_22/ReadVariableOp%batch_normalization_22/ReadVariableOp2R
'batch_normalization_22/ReadVariableOp_1'batch_normalization_22/ReadVariableOp_12p
6batch_normalization_23/FusedBatchNormV3/ReadVariableOp6batch_normalization_23/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_23/FusedBatchNormV3/ReadVariableOp_18batch_normalization_23/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_23/ReadVariableOp%batch_normalization_23/ReadVariableOp2R
'batch_normalization_23/ReadVariableOp_1'batch_normalization_23/ReadVariableOp_12h
2module_wrapper_85/conv2d_41/BiasAdd/ReadVariableOp2module_wrapper_85/conv2d_41/BiasAdd/ReadVariableOp2f
1module_wrapper_85/conv2d_41/Conv2D/ReadVariableOp1module_wrapper_85/conv2d_41/Conv2D/ReadVariableOp2h
2module_wrapper_86/conv2d_42/BiasAdd/ReadVariableOp2module_wrapper_86/conv2d_42/BiasAdd/ReadVariableOp2f
1module_wrapper_86/conv2d_42/Conv2D/ReadVariableOp1module_wrapper_86/conv2d_42/Conv2D/ReadVariableOp2h
2module_wrapper_88/conv2d_43/BiasAdd/ReadVariableOp2module_wrapper_88/conv2d_43/BiasAdd/ReadVariableOp2f
1module_wrapper_88/conv2d_43/Conv2D/ReadVariableOp1module_wrapper_88/conv2d_43/Conv2D/ReadVariableOp2h
2module_wrapper_89/conv2d_44/BiasAdd/ReadVariableOp2module_wrapper_89/conv2d_44/BiasAdd/ReadVariableOp2f
1module_wrapper_89/conv2d_44/Conv2D/ReadVariableOp1module_wrapper_89/conv2d_44/Conv2D/ReadVariableOp2h
2module_wrapper_91/conv2d_45/BiasAdd/ReadVariableOp2module_wrapper_91/conv2d_45/BiasAdd/ReadVariableOp2f
1module_wrapper_91/conv2d_45/Conv2D/ReadVariableOp1module_wrapper_91/conv2d_45/Conv2D/ReadVariableOp2f
1module_wrapper_94/dense_18/BiasAdd/ReadVariableOp1module_wrapper_94/dense_18/BiasAdd/ReadVariableOp2d
0module_wrapper_94/dense_18/MatMul/ReadVariableOp0module_wrapper_94/dense_18/MatMul/ReadVariableOp2f
1module_wrapper_95/dense_19/BiasAdd/ReadVariableOp1module_wrapper_95/dense_19/BiasAdd/ReadVariableOp2d
0module_wrapper_95/dense_19/MatMul/ReadVariableOp0module_wrapper_95/dense_19/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
M__inference_module_wrapper_91_layer_call_and_return_conditional_losses_177923

args_0D
(conv2d_45_conv2d_readvariableop_resource:??8
)conv2d_45_biasadd_readvariableop_resource:	?
identity?? conv2d_45/BiasAdd/ReadVariableOp?conv2d_45/Conv2D/ReadVariableOp?
conv2d_45/Conv2D/ReadVariableOpReadVariableOp(conv2d_45_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
conv2d_45/Conv2DConv2Dargs_0'conv2d_45/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
?
 conv2d_45/BiasAdd/ReadVariableOpReadVariableOp)conv2d_45_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
conv2d_45/BiasAddBiasAddconv2d_45/Conv2D:output:0(conv2d_45/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????m
conv2d_45/ReluReluconv2d_45/BiasAdd:output:0*
T0*0
_output_shapes
:??????????t
IdentityIdentityconv2d_45/Relu:activations:0^NoOp*
T0*0
_output_shapes
:???????????
NoOpNoOp!^conv2d_45/BiasAdd/ReadVariableOp ^conv2d_45/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 2D
 conv2d_45/BiasAdd/ReadVariableOp conv2d_45/BiasAdd/ReadVariableOp2B
conv2d_45/Conv2D/ReadVariableOpconv2d_45/Conv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameargs_0
?
N
2__inference_module_wrapper_93_layer_call_fn_178027

args_0
identity?
PartitionedCallPartitionedCallargs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_93_layer_call_and_return_conditional_losses_176577a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameargs_0
?
?
2__inference_module_wrapper_95_layer_call_fn_178088

args_0
unknown:	?
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_95_layer_call_and_return_conditional_losses_176441o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameargs_0
?
?
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_177691

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
M__inference_module_wrapper_89_layer_call_and_return_conditional_losses_177789

args_0D
(conv2d_44_conv2d_readvariableop_resource:??8
)conv2d_44_biasadd_readvariableop_resource:	?
identity?? conv2d_44/BiasAdd/ReadVariableOp?conv2d_44/Conv2D/ReadVariableOp?
conv2d_44/Conv2D/ReadVariableOpReadVariableOp(conv2d_44_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
conv2d_44/Conv2DConv2Dargs_0'conv2d_44/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
?
 conv2d_44/BiasAdd/ReadVariableOpReadVariableOp)conv2d_44_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
conv2d_44/BiasAddBiasAddconv2d_44/Conv2D:output:0(conv2d_44/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????m
conv2d_44/ReluReluconv2d_44/BiasAdd:output:0*
T0*0
_output_shapes
:??????????t
IdentityIdentityconv2d_44/Relu:activations:0^NoOp*
T0*0
_output_shapes
:???????????
NoOpNoOp!^conv2d_44/BiasAdd/ReadVariableOp ^conv2d_44/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????

?: : 2D
 conv2d_44/BiasAdd/ReadVariableOp conv2d_44/BiasAdd/ReadVariableOp2B
conv2d_44/Conv2D/ReadVariableOpconv2d_44/Conv2D/ReadVariableOp:X T
0
_output_shapes
:?????????

?
 
_user_specified_nameargs_0
?
h
L__inference_max_pooling2d_26_layer_call_and_return_conditional_losses_177641

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
M__inference_module_wrapper_85_layer_call_and_return_conditional_losses_177574

args_0B
(conv2d_41_conv2d_readvariableop_resource:@7
)conv2d_41_biasadd_readvariableop_resource:@
identity?? conv2d_41/BiasAdd/ReadVariableOp?conv2d_41/Conv2D/ReadVariableOp?
conv2d_41/Conv2D/ReadVariableOpReadVariableOp(conv2d_41_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
conv2d_41/Conv2DConv2Dargs_0'conv2d_41/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
?
 conv2d_41/BiasAdd/ReadVariableOpReadVariableOp)conv2d_41_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_41/BiasAddBiasAddconv2d_41/Conv2D:output:0(conv2d_41/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@l
conv2d_41/ReluReluconv2d_41/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@s
IdentityIdentityconv2d_41/Relu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@?
NoOpNoOp!^conv2d_41/BiasAdd/ReadVariableOp ^conv2d_41/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 2D
 conv2d_41/BiasAdd/ReadVariableOp conv2d_41/BiasAdd/ReadVariableOp2B
conv2d_41/Conv2D/ReadVariableOpconv2d_41/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameargs_0
?
?
M__inference_module_wrapper_94_layer_call_and_return_conditional_losses_178079

args_0;
'dense_18_matmul_readvariableop_resource:
??7
(dense_18_biasadd_readvariableop_resource:	?
identity??dense_18/BiasAdd/ReadVariableOp?dense_18/MatMul/ReadVariableOp?
dense_18/MatMul/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0|
dense_18/MatMulMatMulargs_0&dense_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_18/BiasAdd/ReadVariableOpReadVariableOp(dense_18_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_18/BiasAddBiasAdddense_18/MatMul:product:0'dense_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????c
dense_18/ReluReludense_18/BiasAdd:output:0*
T0*(
_output_shapes
:??????????k
IdentityIdentitydense_18/Relu:activations:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 2B
dense_18/BiasAdd/ReadVariableOpdense_18/BiasAdd/ReadVariableOp2@
dense_18/MatMul/ReadVariableOpdense_18/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameargs_0
?
i
M__inference_module_wrapper_87_layer_call_and_return_conditional_losses_177635

args_0
identity?
max_pooling2d_26/MaxPoolMaxPoolargs_0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
q
IdentityIdentity!max_pooling2d_26/MaxPool:output:0*
T0*/
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameargs_0
??
?&
__inference__traced_save_178358
file_prefix;
7savev2_batch_normalization_21_gamma_read_readvariableop:
6savev2_batch_normalization_21_beta_read_readvariableopA
=savev2_batch_normalization_21_moving_mean_read_readvariableopE
Asavev2_batch_normalization_21_moving_variance_read_readvariableop;
7savev2_batch_normalization_22_gamma_read_readvariableop:
6savev2_batch_normalization_22_beta_read_readvariableopA
=savev2_batch_normalization_22_moving_mean_read_readvariableopE
Asavev2_batch_normalization_22_moving_variance_read_readvariableop;
7savev2_batch_normalization_23_gamma_read_readvariableop:
6savev2_batch_normalization_23_beta_read_readvariableopA
=savev2_batch_normalization_23_moving_mean_read_readvariableopE
Asavev2_batch_normalization_23_moving_variance_read_readvariableopA
=savev2_module_wrapper_85_conv2d_41_kernel_read_readvariableop?
;savev2_module_wrapper_85_conv2d_41_bias_read_readvariableopA
=savev2_module_wrapper_86_conv2d_42_kernel_read_readvariableop?
;savev2_module_wrapper_86_conv2d_42_bias_read_readvariableopA
=savev2_module_wrapper_88_conv2d_43_kernel_read_readvariableop?
;savev2_module_wrapper_88_conv2d_43_bias_read_readvariableopA
=savev2_module_wrapper_89_conv2d_44_kernel_read_readvariableop?
;savev2_module_wrapper_89_conv2d_44_bias_read_readvariableopA
=savev2_module_wrapper_91_conv2d_45_kernel_read_readvariableop?
;savev2_module_wrapper_91_conv2d_45_bias_read_readvariableop@
<savev2_module_wrapper_94_dense_18_kernel_read_readvariableop>
:savev2_module_wrapper_94_dense_18_bias_read_readvariableop@
<savev2_module_wrapper_95_dense_19_kernel_read_readvariableop>
:savev2_module_wrapper_95_dense_19_bias_read_readvariableop(
$savev2_iteration_read_readvariableop	,
(savev2_learning_rate_read_readvariableopH
Dsavev2_adam_m_module_wrapper_85_conv2d_41_kernel_read_readvariableopH
Dsavev2_adam_v_module_wrapper_85_conv2d_41_kernel_read_readvariableopF
Bsavev2_adam_m_module_wrapper_85_conv2d_41_bias_read_readvariableopF
Bsavev2_adam_v_module_wrapper_85_conv2d_41_bias_read_readvariableopH
Dsavev2_adam_m_module_wrapper_86_conv2d_42_kernel_read_readvariableopH
Dsavev2_adam_v_module_wrapper_86_conv2d_42_kernel_read_readvariableopF
Bsavev2_adam_m_module_wrapper_86_conv2d_42_bias_read_readvariableopF
Bsavev2_adam_v_module_wrapper_86_conv2d_42_bias_read_readvariableopB
>savev2_adam_m_batch_normalization_21_gamma_read_readvariableopB
>savev2_adam_v_batch_normalization_21_gamma_read_readvariableopA
=savev2_adam_m_batch_normalization_21_beta_read_readvariableopA
=savev2_adam_v_batch_normalization_21_beta_read_readvariableopH
Dsavev2_adam_m_module_wrapper_88_conv2d_43_kernel_read_readvariableopH
Dsavev2_adam_v_module_wrapper_88_conv2d_43_kernel_read_readvariableopF
Bsavev2_adam_m_module_wrapper_88_conv2d_43_bias_read_readvariableopF
Bsavev2_adam_v_module_wrapper_88_conv2d_43_bias_read_readvariableopH
Dsavev2_adam_m_module_wrapper_89_conv2d_44_kernel_read_readvariableopH
Dsavev2_adam_v_module_wrapper_89_conv2d_44_kernel_read_readvariableopF
Bsavev2_adam_m_module_wrapper_89_conv2d_44_bias_read_readvariableopF
Bsavev2_adam_v_module_wrapper_89_conv2d_44_bias_read_readvariableopB
>savev2_adam_m_batch_normalization_22_gamma_read_readvariableopB
>savev2_adam_v_batch_normalization_22_gamma_read_readvariableopA
=savev2_adam_m_batch_normalization_22_beta_read_readvariableopA
=savev2_adam_v_batch_normalization_22_beta_read_readvariableopH
Dsavev2_adam_m_module_wrapper_91_conv2d_45_kernel_read_readvariableopH
Dsavev2_adam_v_module_wrapper_91_conv2d_45_kernel_read_readvariableopF
Bsavev2_adam_m_module_wrapper_91_conv2d_45_bias_read_readvariableopF
Bsavev2_adam_v_module_wrapper_91_conv2d_45_bias_read_readvariableopB
>savev2_adam_m_batch_normalization_23_gamma_read_readvariableopB
>savev2_adam_v_batch_normalization_23_gamma_read_readvariableopA
=savev2_adam_m_batch_normalization_23_beta_read_readvariableopA
=savev2_adam_v_batch_normalization_23_beta_read_readvariableopG
Csavev2_adam_m_module_wrapper_94_dense_18_kernel_read_readvariableopG
Csavev2_adam_v_module_wrapper_94_dense_18_kernel_read_readvariableopE
Asavev2_adam_m_module_wrapper_94_dense_18_bias_read_readvariableopE
Asavev2_adam_v_module_wrapper_94_dense_18_bias_read_readvariableopG
Csavev2_adam_m_module_wrapper_95_dense_19_kernel_read_readvariableopG
Csavev2_adam_v_module_wrapper_95_dense_19_kernel_read_readvariableopE
Asavev2_adam_m_module_wrapper_95_dense_19_bias_read_readvariableopE
Asavev2_adam_v_module_wrapper_95_dense_19_bias_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
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
_temp/part?
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
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:I*
dtype0*?
value?B?IB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:I*
dtype0*?
value?B?IB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?%
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:07savev2_batch_normalization_21_gamma_read_readvariableop6savev2_batch_normalization_21_beta_read_readvariableop=savev2_batch_normalization_21_moving_mean_read_readvariableopAsavev2_batch_normalization_21_moving_variance_read_readvariableop7savev2_batch_normalization_22_gamma_read_readvariableop6savev2_batch_normalization_22_beta_read_readvariableop=savev2_batch_normalization_22_moving_mean_read_readvariableopAsavev2_batch_normalization_22_moving_variance_read_readvariableop7savev2_batch_normalization_23_gamma_read_readvariableop6savev2_batch_normalization_23_beta_read_readvariableop=savev2_batch_normalization_23_moving_mean_read_readvariableopAsavev2_batch_normalization_23_moving_variance_read_readvariableop=savev2_module_wrapper_85_conv2d_41_kernel_read_readvariableop;savev2_module_wrapper_85_conv2d_41_bias_read_readvariableop=savev2_module_wrapper_86_conv2d_42_kernel_read_readvariableop;savev2_module_wrapper_86_conv2d_42_bias_read_readvariableop=savev2_module_wrapper_88_conv2d_43_kernel_read_readvariableop;savev2_module_wrapper_88_conv2d_43_bias_read_readvariableop=savev2_module_wrapper_89_conv2d_44_kernel_read_readvariableop;savev2_module_wrapper_89_conv2d_44_bias_read_readvariableop=savev2_module_wrapper_91_conv2d_45_kernel_read_readvariableop;savev2_module_wrapper_91_conv2d_45_bias_read_readvariableop<savev2_module_wrapper_94_dense_18_kernel_read_readvariableop:savev2_module_wrapper_94_dense_18_bias_read_readvariableop<savev2_module_wrapper_95_dense_19_kernel_read_readvariableop:savev2_module_wrapper_95_dense_19_bias_read_readvariableop$savev2_iteration_read_readvariableop(savev2_learning_rate_read_readvariableopDsavev2_adam_m_module_wrapper_85_conv2d_41_kernel_read_readvariableopDsavev2_adam_v_module_wrapper_85_conv2d_41_kernel_read_readvariableopBsavev2_adam_m_module_wrapper_85_conv2d_41_bias_read_readvariableopBsavev2_adam_v_module_wrapper_85_conv2d_41_bias_read_readvariableopDsavev2_adam_m_module_wrapper_86_conv2d_42_kernel_read_readvariableopDsavev2_adam_v_module_wrapper_86_conv2d_42_kernel_read_readvariableopBsavev2_adam_m_module_wrapper_86_conv2d_42_bias_read_readvariableopBsavev2_adam_v_module_wrapper_86_conv2d_42_bias_read_readvariableop>savev2_adam_m_batch_normalization_21_gamma_read_readvariableop>savev2_adam_v_batch_normalization_21_gamma_read_readvariableop=savev2_adam_m_batch_normalization_21_beta_read_readvariableop=savev2_adam_v_batch_normalization_21_beta_read_readvariableopDsavev2_adam_m_module_wrapper_88_conv2d_43_kernel_read_readvariableopDsavev2_adam_v_module_wrapper_88_conv2d_43_kernel_read_readvariableopBsavev2_adam_m_module_wrapper_88_conv2d_43_bias_read_readvariableopBsavev2_adam_v_module_wrapper_88_conv2d_43_bias_read_readvariableopDsavev2_adam_m_module_wrapper_89_conv2d_44_kernel_read_readvariableopDsavev2_adam_v_module_wrapper_89_conv2d_44_kernel_read_readvariableopBsavev2_adam_m_module_wrapper_89_conv2d_44_bias_read_readvariableopBsavev2_adam_v_module_wrapper_89_conv2d_44_bias_read_readvariableop>savev2_adam_m_batch_normalization_22_gamma_read_readvariableop>savev2_adam_v_batch_normalization_22_gamma_read_readvariableop=savev2_adam_m_batch_normalization_22_beta_read_readvariableop=savev2_adam_v_batch_normalization_22_beta_read_readvariableopDsavev2_adam_m_module_wrapper_91_conv2d_45_kernel_read_readvariableopDsavev2_adam_v_module_wrapper_91_conv2d_45_kernel_read_readvariableopBsavev2_adam_m_module_wrapper_91_conv2d_45_bias_read_readvariableopBsavev2_adam_v_module_wrapper_91_conv2d_45_bias_read_readvariableop>savev2_adam_m_batch_normalization_23_gamma_read_readvariableop>savev2_adam_v_batch_normalization_23_gamma_read_readvariableop=savev2_adam_m_batch_normalization_23_beta_read_readvariableop=savev2_adam_v_batch_normalization_23_beta_read_readvariableopCsavev2_adam_m_module_wrapper_94_dense_18_kernel_read_readvariableopCsavev2_adam_v_module_wrapper_94_dense_18_kernel_read_readvariableopAsavev2_adam_m_module_wrapper_94_dense_18_bias_read_readvariableopAsavev2_adam_v_module_wrapper_94_dense_18_bias_read_readvariableopCsavev2_adam_m_module_wrapper_95_dense_19_kernel_read_readvariableopCsavev2_adam_v_module_wrapper_95_dense_19_kernel_read_readvariableopAsavev2_adam_m_module_wrapper_95_dense_19_bias_read_readvariableopAsavev2_adam_v_module_wrapper_95_dense_19_bias_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *W
dtypesM
K2I	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
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

identity_1Identity_1:output:0*?
_input_shapes?
?: :@:@:@:@:?:?:?:?:?:?:?:?:@:@:@@:@:@?:?:??:?:??:?:
??:?:	?:: : :@:@:@:@:@@:@@:@:@:@:@:@:@:@?:@?:?:?:??:??:?:?:?:?:?:?:??:??:?:?:?:?:?:?:
??:
??:?:?:	?:	?::: : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!	

_output_shapes	
:?:!


_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:,(
&
_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:-)
'
_output_shapes
:@?:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:@:,(
&
_output_shapes
:@: 

_output_shapes
:@:  

_output_shapes
:@:,!(
&
_output_shapes
:@@:,"(
&
_output_shapes
:@@: #

_output_shapes
:@: $

_output_shapes
:@: %

_output_shapes
:@: &

_output_shapes
:@: '

_output_shapes
:@: (

_output_shapes
:@:-))
'
_output_shapes
:@?:-*)
'
_output_shapes
:@?:!+

_output_shapes	
:?:!,

_output_shapes	
:?:.-*
(
_output_shapes
:??:..*
(
_output_shapes
:??:!/

_output_shapes	
:?:!0

_output_shapes	
:?:!1

_output_shapes	
:?:!2

_output_shapes	
:?:!3

_output_shapes	
:?:!4

_output_shapes	
:?:.5*
(
_output_shapes
:??:.6*
(
_output_shapes
:??:!7

_output_shapes	
:?:!8

_output_shapes	
:?:!9

_output_shapes	
:?:!:

_output_shapes	
:?:!;

_output_shapes	
:?:!<

_output_shapes	
:?:&="
 
_output_shapes
:
??:&>"
 
_output_shapes
:
??:!?

_output_shapes	
:?:!@

_output_shapes	
:?:%A!

_output_shapes
:	?:%B!

_output_shapes
:	?: C

_output_shapes
:: D

_output_shapes
::E

_output_shapes
: :F

_output_shapes
: :G

_output_shapes
: :H

_output_shapes
: :I

_output_shapes
: 
?
?
2__inference_module_wrapper_86_layer_call_fn_177584

args_0!
unknown:@@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_86_layer_call_and_return_conditional_losses_176300w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameargs_0
?	
?
7__inference_batch_normalization_23_layer_call_fn_177968

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_176223?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
M__inference_module_wrapper_91_layer_call_and_return_conditional_losses_177912

args_0D
(conv2d_45_conv2d_readvariableop_resource:??8
)conv2d_45_biasadd_readvariableop_resource:	?
identity?? conv2d_45/BiasAdd/ReadVariableOp?conv2d_45/Conv2D/ReadVariableOp?
conv2d_45/Conv2D/ReadVariableOpReadVariableOp(conv2d_45_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
conv2d_45/Conv2DConv2Dargs_0'conv2d_45/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
?
 conv2d_45/BiasAdd/ReadVariableOpReadVariableOp)conv2d_45_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
conv2d_45/BiasAddBiasAddconv2d_45/Conv2D:output:0(conv2d_45/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????m
conv2d_45/ReluReluconv2d_45/BiasAdd:output:0*
T0*0
_output_shapes
:??????????t
IdentityIdentityconv2d_45/Relu:activations:0^NoOp*
T0*0
_output_shapes
:???????????
NoOpNoOp!^conv2d_45/BiasAdd/ReadVariableOp ^conv2d_45/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 2D
 conv2d_45/BiasAdd/ReadVariableOp conv2d_45/BiasAdd/ReadVariableOp2B
conv2d_45/Conv2D/ReadVariableOpconv2d_45/Conv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameargs_0
?M
?
H__inference_sequential_9_layer_call_and_return_conditional_losses_177089
input_52
module_wrapper_85_177022:@&
module_wrapper_85_177024:@2
module_wrapper_86_177027:@@&
module_wrapper_86_177029:@+
batch_normalization_21_177033:@+
batch_normalization_21_177035:@+
batch_normalization_21_177037:@+
batch_normalization_21_177039:@3
module_wrapper_88_177042:@?'
module_wrapper_88_177044:	?4
module_wrapper_89_177047:??'
module_wrapper_89_177049:	?,
batch_normalization_22_177053:	?,
batch_normalization_22_177055:	?,
batch_normalization_22_177057:	?,
batch_normalization_22_177059:	?4
module_wrapper_91_177062:??'
module_wrapper_91_177064:	?,
batch_normalization_23_177068:	?,
batch_normalization_23_177070:	?,
batch_normalization_23_177072:	?,
batch_normalization_23_177074:	?,
module_wrapper_94_177078:
??'
module_wrapper_94_177080:	?+
module_wrapper_95_177083:	?&
module_wrapper_95_177085:
identity??.batch_normalization_21/StatefulPartitionedCall?.batch_normalization_22/StatefulPartitionedCall?.batch_normalization_23/StatefulPartitionedCall?)module_wrapper_85/StatefulPartitionedCall?)module_wrapper_86/StatefulPartitionedCall?)module_wrapper_88/StatefulPartitionedCall?)module_wrapper_89/StatefulPartitionedCall?)module_wrapper_91/StatefulPartitionedCall?)module_wrapper_94/StatefulPartitionedCall?)module_wrapper_95/StatefulPartitionedCall?
)module_wrapper_85/StatefulPartitionedCallStatefulPartitionedCallinput_5module_wrapper_85_177022module_wrapper_85_177024*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_85_layer_call_and_return_conditional_losses_176283?
)module_wrapper_86/StatefulPartitionedCallStatefulPartitionedCall2module_wrapper_85/StatefulPartitionedCall:output:0module_wrapper_86_177027module_wrapper_86_177029*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_86_layer_call_and_return_conditional_losses_176300?
!module_wrapper_87/PartitionedCallPartitionedCall2module_wrapper_86/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_87_layer_call_and_return_conditional_losses_176311?
.batch_normalization_21/StatefulPartitionedCallStatefulPartitionedCall*module_wrapper_87/PartitionedCall:output:0batch_normalization_21_177033batch_normalization_21_177035batch_normalization_21_177037batch_normalization_21_177039*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_176095?
)module_wrapper_88/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_21/StatefulPartitionedCall:output:0module_wrapper_88_177042module_wrapper_88_177044*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????

?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_88_layer_call_and_return_conditional_losses_176333?
)module_wrapper_89/StatefulPartitionedCallStatefulPartitionedCall2module_wrapper_88/StatefulPartitionedCall:output:0module_wrapper_89_177047module_wrapper_89_177049*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_89_layer_call_and_return_conditional_losses_176350?
!module_wrapper_90/PartitionedCallPartitionedCall2module_wrapper_89/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_90_layer_call_and_return_conditional_losses_176361?
.batch_normalization_22/StatefulPartitionedCallStatefulPartitionedCall*module_wrapper_90/PartitionedCall:output:0batch_normalization_22_177053batch_normalization_22_177055batch_normalization_22_177057batch_normalization_22_177059*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_176159?
)module_wrapper_91/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_22/StatefulPartitionedCall:output:0module_wrapper_91_177062module_wrapper_91_177064*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_91_layer_call_and_return_conditional_losses_176383?
!module_wrapper_92/PartitionedCallPartitionedCall2module_wrapper_91/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_92_layer_call_and_return_conditional_losses_176394?
.batch_normalization_23/StatefulPartitionedCallStatefulPartitionedCall*module_wrapper_92/PartitionedCall:output:0batch_normalization_23_177068batch_normalization_23_177070batch_normalization_23_177072batch_normalization_23_177074*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_176223?
!module_wrapper_93/PartitionedCallPartitionedCall7batch_normalization_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_93_layer_call_and_return_conditional_losses_176411?
)module_wrapper_94/StatefulPartitionedCallStatefulPartitionedCall*module_wrapper_93/PartitionedCall:output:0module_wrapper_94_177078module_wrapper_94_177080*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_94_layer_call_and_return_conditional_losses_176424?
)module_wrapper_95/StatefulPartitionedCallStatefulPartitionedCall2module_wrapper_94/StatefulPartitionedCall:output:0module_wrapper_95_177083module_wrapper_95_177085*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_95_layer_call_and_return_conditional_losses_176441?
IdentityIdentity2module_wrapper_95/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp/^batch_normalization_21/StatefulPartitionedCall/^batch_normalization_22/StatefulPartitionedCall/^batch_normalization_23/StatefulPartitionedCall*^module_wrapper_85/StatefulPartitionedCall*^module_wrapper_86/StatefulPartitionedCall*^module_wrapper_88/StatefulPartitionedCall*^module_wrapper_89/StatefulPartitionedCall*^module_wrapper_91/StatefulPartitionedCall*^module_wrapper_94/StatefulPartitionedCall*^module_wrapper_95/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_21/StatefulPartitionedCall.batch_normalization_21/StatefulPartitionedCall2`
.batch_normalization_22/StatefulPartitionedCall.batch_normalization_22/StatefulPartitionedCall2`
.batch_normalization_23/StatefulPartitionedCall.batch_normalization_23/StatefulPartitionedCall2V
)module_wrapper_85/StatefulPartitionedCall)module_wrapper_85/StatefulPartitionedCall2V
)module_wrapper_86/StatefulPartitionedCall)module_wrapper_86/StatefulPartitionedCall2V
)module_wrapper_88/StatefulPartitionedCall)module_wrapper_88/StatefulPartitionedCall2V
)module_wrapper_89/StatefulPartitionedCall)module_wrapper_89/StatefulPartitionedCall2V
)module_wrapper_91/StatefulPartitionedCall)module_wrapper_91/StatefulPartitionedCall2V
)module_wrapper_94/StatefulPartitionedCall)module_wrapper_94/StatefulPartitionedCall2V
)module_wrapper_95/StatefulPartitionedCall)module_wrapper_95/StatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_5
?
?
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_177709

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
-__inference_sequential_9_layer_call_fn_177019
input_5!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@
	unknown_6:@$
	unknown_7:@?
	unknown_8:	?%
	unknown_9:??

unknown_10:	?

unknown_11:	?

unknown_12:	?

unknown_13:	?

unknown_14:	?&

unknown_15:??

unknown_16:	?

unknown_17:	?

unknown_18:	?

unknown_19:	?

unknown_20:	?

unknown_21:
??

unknown_22:	?

unknown_23:	?

unknown_24:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_9_layer_call_and_return_conditional_losses_176907o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_5
?
M
1__inference_max_pooling2d_28_layer_call_fn_177955

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_28_layer_call_and_return_conditional_losses_177949?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_176190

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
h
L__inference_max_pooling2d_28_layer_call_and_return_conditional_losses_177949

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_177999

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_176223

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
N
2__inference_module_wrapper_87_layer_call_fn_177625

args_0
identity?
PartitionedCallPartitionedCallargs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_87_layer_call_and_return_conditional_losses_176715h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameargs_0
?
?
2__inference_module_wrapper_94_layer_call_fn_178057

args_0
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_94_layer_call_and_return_conditional_losses_176556p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameargs_0
?
?
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_176254

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
M__inference_module_wrapper_89_layer_call_and_return_conditional_losses_177778

args_0D
(conv2d_44_conv2d_readvariableop_resource:??8
)conv2d_44_biasadd_readvariableop_resource:	?
identity?? conv2d_44/BiasAdd/ReadVariableOp?conv2d_44/Conv2D/ReadVariableOp?
conv2d_44/Conv2D/ReadVariableOpReadVariableOp(conv2d_44_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
conv2d_44/Conv2DConv2Dargs_0'conv2d_44/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
?
 conv2d_44/BiasAdd/ReadVariableOpReadVariableOp)conv2d_44_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
conv2d_44/BiasAddBiasAddconv2d_44/Conv2D:output:0(conv2d_44/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????m
conv2d_44/ReluReluconv2d_44/BiasAdd:output:0*
T0*0
_output_shapes
:??????????t
IdentityIdentityconv2d_44/Relu:activations:0^NoOp*
T0*0
_output_shapes
:???????????
NoOpNoOp!^conv2d_44/BiasAdd/ReadVariableOp ^conv2d_44/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????

?: : 2D
 conv2d_44/BiasAdd/ReadVariableOp conv2d_44/BiasAdd/ReadVariableOp2B
conv2d_44/Conv2D/ReadVariableOpconv2d_44/Conv2D/ReadVariableOp:X T
0
_output_shapes
:?????????

?
 
_user_specified_nameargs_0
?
i
M__inference_module_wrapper_90_layer_call_and_return_conditional_losses_177804

args_0
identity?
max_pooling2d_27/MaxPoolMaxPoolargs_0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
r
IdentityIdentity!max_pooling2d_27/MaxPool:output:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameargs_0
?
?
M__inference_module_wrapper_91_layer_call_and_return_conditional_losses_176619

args_0D
(conv2d_45_conv2d_readvariableop_resource:??8
)conv2d_45_biasadd_readvariableop_resource:	?
identity?? conv2d_45/BiasAdd/ReadVariableOp?conv2d_45/Conv2D/ReadVariableOp?
conv2d_45/Conv2D/ReadVariableOpReadVariableOp(conv2d_45_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
conv2d_45/Conv2DConv2Dargs_0'conv2d_45/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
?
 conv2d_45/BiasAdd/ReadVariableOpReadVariableOp)conv2d_45_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
conv2d_45/BiasAddBiasAddconv2d_45/Conv2D:output:0(conv2d_45/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????m
conv2d_45/ReluReluconv2d_45/BiasAdd:output:0*
T0*0
_output_shapes
:??????????t
IdentityIdentityconv2d_45/Relu:activations:0^NoOp*
T0*0
_output_shapes
:???????????
NoOpNoOp!^conv2d_45/BiasAdd/ReadVariableOp ^conv2d_45/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 2D
 conv2d_45/BiasAdd/ReadVariableOp conv2d_45/BiasAdd/ReadVariableOp2B
conv2d_45/Conv2D/ReadVariableOpconv2d_45/Conv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameargs_0
?	
?
7__inference_batch_normalization_22_layer_call_fn_177847

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_176190?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
M__inference_module_wrapper_94_layer_call_and_return_conditional_losses_176424

args_0;
'dense_18_matmul_readvariableop_resource:
??7
(dense_18_biasadd_readvariableop_resource:	?
identity??dense_18/BiasAdd/ReadVariableOp?dense_18/MatMul/ReadVariableOp?
dense_18/MatMul/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0|
dense_18/MatMulMatMulargs_0&dense_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_18/BiasAdd/ReadVariableOpReadVariableOp(dense_18_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_18/BiasAddBiasAdddense_18/MatMul:product:0'dense_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????c
dense_18/ReluReludense_18/BiasAdd:output:0*
T0*(
_output_shapes
:??????????k
IdentityIdentitydense_18/Relu:activations:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 2B
dense_18/BiasAdd/ReadVariableOpdense_18/BiasAdd/ReadVariableOp2@
dense_18/MatMul/ReadVariableOpdense_18/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameargs_0
?
?
M__inference_module_wrapper_88_layer_call_and_return_conditional_losses_177749

args_0C
(conv2d_43_conv2d_readvariableop_resource:@?8
)conv2d_43_biasadd_readvariableop_resource:	?
identity?? conv2d_43/BiasAdd/ReadVariableOp?conv2d_43/Conv2D/ReadVariableOp?
conv2d_43/Conv2D/ReadVariableOpReadVariableOp(conv2d_43_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
conv2d_43/Conv2DConv2Dargs_0'conv2d_43/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????

?*
paddingVALID*
strides
?
 conv2d_43/BiasAdd/ReadVariableOpReadVariableOp)conv2d_43_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
conv2d_43/BiasAddBiasAddconv2d_43/Conv2D:output:0(conv2d_43/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????

?m
conv2d_43/ReluReluconv2d_43/BiasAdd:output:0*
T0*0
_output_shapes
:?????????

?t
IdentityIdentityconv2d_43/Relu:activations:0^NoOp*
T0*0
_output_shapes
:?????????

??
NoOpNoOp!^conv2d_43/BiasAdd/ReadVariableOp ^conv2d_43/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 2D
 conv2d_43/BiasAdd/ReadVariableOp conv2d_43/BiasAdd/ReadVariableOp2B
conv2d_43/Conv2D/ReadVariableOpconv2d_43/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameargs_0
?
i
M__inference_module_wrapper_90_layer_call_and_return_conditional_losses_176361

args_0
identity?
max_pooling2d_27/MaxPoolMaxPoolargs_0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
r
IdentityIdentity!max_pooling2d_27/MaxPool:output:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameargs_0
?
N
2__inference_module_wrapper_90_layer_call_fn_177799

args_0
identity?
PartitionedCallPartitionedCallargs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_90_layer_call_and_return_conditional_losses_176639i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameargs_0
?
?
2__inference_module_wrapper_86_layer_call_fn_177593

args_0!
unknown:@@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_86_layer_call_and_return_conditional_losses_176741w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameargs_0
?
?
2__inference_module_wrapper_89_layer_call_fn_177758

args_0#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_89_layer_call_and_return_conditional_losses_176350x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????

?: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????

?
 
_user_specified_nameargs_0
?	
?
7__inference_batch_normalization_21_layer_call_fn_177660

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_176095?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_176159

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?	
?
7__inference_batch_normalization_21_layer_call_fn_177673

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_176126?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_178017

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
i
M__inference_module_wrapper_92_layer_call_and_return_conditional_losses_176593

args_0
identity?
max_pooling2d_28/MaxPoolMaxPoolargs_0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
r
IdentityIdentity!max_pooling2d_28/MaxPool:output:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameargs_0
?
i
M__inference_module_wrapper_93_layer_call_and_return_conditional_losses_176411

args_0
identity`
flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   q
flatten_9/ReshapeReshapeargs_0flatten_9/Const:output:0*
T0*(
_output_shapes
:??????????c
IdentityIdentityflatten_9/Reshape:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameargs_0
?
?
M__inference_module_wrapper_89_layer_call_and_return_conditional_losses_176665

args_0D
(conv2d_44_conv2d_readvariableop_resource:??8
)conv2d_44_biasadd_readvariableop_resource:	?
identity?? conv2d_44/BiasAdd/ReadVariableOp?conv2d_44/Conv2D/ReadVariableOp?
conv2d_44/Conv2D/ReadVariableOpReadVariableOp(conv2d_44_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
conv2d_44/Conv2DConv2Dargs_0'conv2d_44/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
?
 conv2d_44/BiasAdd/ReadVariableOpReadVariableOp)conv2d_44_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
conv2d_44/BiasAddBiasAddconv2d_44/Conv2D:output:0(conv2d_44/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????m
conv2d_44/ReluReluconv2d_44/BiasAdd:output:0*
T0*0
_output_shapes
:??????????t
IdentityIdentityconv2d_44/Relu:activations:0^NoOp*
T0*0
_output_shapes
:???????????
NoOpNoOp!^conv2d_44/BiasAdd/ReadVariableOp ^conv2d_44/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????

?: : 2D
 conv2d_44/BiasAdd/ReadVariableOp conv2d_44/BiasAdd/ReadVariableOp2B
conv2d_44/Conv2D/ReadVariableOpconv2d_44/Conv2D/ReadVariableOp:X T
0
_output_shapes
:?????????

?
 
_user_specified_nameargs_0
?
?
M__inference_module_wrapper_95_layer_call_and_return_conditional_losses_178119

args_0:
'dense_19_matmul_readvariableop_resource:	?6
(dense_19_biasadd_readvariableop_resource:
identity??dense_19/BiasAdd/ReadVariableOp?dense_19/MatMul/ReadVariableOp?
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0{
dense_19/MatMulMatMulargs_0&dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_19/BiasAdd/ReadVariableOpReadVariableOp(dense_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_19/BiasAddBiasAdddense_19/MatMul:product:0'dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????h
dense_19/SoftmaxSoftmaxdense_19/BiasAdd:output:0*
T0*'
_output_shapes
:?????????i
IdentityIdentitydense_19/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 2B
dense_19/BiasAdd/ReadVariableOpdense_19/BiasAdd/ReadVariableOp2@
dense_19/MatMul/ReadVariableOpdense_19/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameargs_0
?
?
2__inference_module_wrapper_91_layer_call_fn_177892

args_0#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_91_layer_call_and_return_conditional_losses_176383x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameargs_0
?	
?
7__inference_batch_normalization_22_layer_call_fn_177834

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_176159?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
i
M__inference_module_wrapper_87_layer_call_and_return_conditional_losses_176715

args_0
identity?
max_pooling2d_26/MaxPoolMaxPoolargs_0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
q
IdentityIdentity!max_pooling2d_26/MaxPool:output:0*
T0*/
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameargs_0
?
N
2__inference_module_wrapper_90_layer_call_fn_177794

args_0
identity?
PartitionedCallPartitionedCallargs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_90_layer_call_and_return_conditional_losses_176361i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameargs_0
?
?
2__inference_module_wrapper_91_layer_call_fn_177901

args_0#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_91_layer_call_and_return_conditional_losses_176619x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameargs_0
?
i
M__inference_module_wrapper_93_layer_call_and_return_conditional_losses_176577

args_0
identity`
flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   q
flatten_9/ReshapeReshapeargs_0flatten_9/Const:output:0*
T0*(
_output_shapes
:??????????c
IdentityIdentityflatten_9/Reshape:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameargs_0
?	
?
7__inference_batch_normalization_23_layer_call_fn_177981

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_176254?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
2__inference_module_wrapper_85_layer_call_fn_177543

args_0!
unknown:@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_85_layer_call_and_return_conditional_losses_176283w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameargs_0
?
i
M__inference_module_wrapper_87_layer_call_and_return_conditional_losses_176311

args_0
identity?
max_pooling2d_26/MaxPoolMaxPoolargs_0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
q
IdentityIdentity!max_pooling2d_26/MaxPool:output:0*
T0*/
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameargs_0
?
?
M__inference_module_wrapper_85_layer_call_and_return_conditional_losses_176283

args_0B
(conv2d_41_conv2d_readvariableop_resource:@7
)conv2d_41_biasadd_readvariableop_resource:@
identity?? conv2d_41/BiasAdd/ReadVariableOp?conv2d_41/Conv2D/ReadVariableOp?
conv2d_41/Conv2D/ReadVariableOpReadVariableOp(conv2d_41_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
conv2d_41/Conv2DConv2Dargs_0'conv2d_41/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
?
 conv2d_41/BiasAdd/ReadVariableOpReadVariableOp)conv2d_41_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_41/BiasAddBiasAddconv2d_41/Conv2D:output:0(conv2d_41/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@l
conv2d_41/ReluReluconv2d_41/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@s
IdentityIdentityconv2d_41/Relu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@?
NoOpNoOp!^conv2d_41/BiasAdd/ReadVariableOp ^conv2d_41/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 2D
 conv2d_41/BiasAdd/ReadVariableOp conv2d_41/BiasAdd/ReadVariableOp2B
conv2d_41/Conv2D/ReadVariableOpconv2d_41/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameargs_0
?
i
M__inference_module_wrapper_92_layer_call_and_return_conditional_losses_177938

args_0
identity?
max_pooling2d_28/MaxPoolMaxPoolargs_0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
r
IdentityIdentity!max_pooling2d_28/MaxPool:output:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameargs_0
?
i
M__inference_module_wrapper_87_layer_call_and_return_conditional_losses_177630

args_0
identity?
max_pooling2d_26/MaxPoolMaxPoolargs_0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
q
IdentityIdentity!max_pooling2d_26/MaxPool:output:0*
T0*/
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameargs_0
?
?
M__inference_module_wrapper_95_layer_call_and_return_conditional_losses_176526

args_0:
'dense_19_matmul_readvariableop_resource:	?6
(dense_19_biasadd_readvariableop_resource:
identity??dense_19/BiasAdd/ReadVariableOp?dense_19/MatMul/ReadVariableOp?
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0{
dense_19/MatMulMatMulargs_0&dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_19/BiasAdd/ReadVariableOpReadVariableOp(dense_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_19/BiasAddBiasAdddense_19/MatMul:product:0'dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????h
dense_19/SoftmaxSoftmaxdense_19/BiasAdd:output:0*
T0*'
_output_shapes
:?????????i
IdentityIdentitydense_19/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 2B
dense_19/BiasAdd/ReadVariableOpdense_19/BiasAdd/ReadVariableOp2@
dense_19/MatMul/ReadVariableOpdense_19/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameargs_0
?
?
2__inference_module_wrapper_95_layer_call_fn_178097

args_0
unknown:	?
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_95_layer_call_and_return_conditional_losses_176526o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameargs_0
?
i
M__inference_module_wrapper_90_layer_call_and_return_conditional_losses_176639

args_0
identity?
max_pooling2d_27/MaxPoolMaxPoolargs_0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
r
IdentityIdentity!max_pooling2d_27/MaxPool:output:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameargs_0
?
M
1__inference_max_pooling2d_26_layer_call_fn_177647

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_26_layer_call_and_return_conditional_losses_177641?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
2__inference_module_wrapper_94_layer_call_fn_178048

args_0
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_94_layer_call_and_return_conditional_losses_176424p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameargs_0
?
?
-__inference_sequential_9_layer_call_fn_176503
input_5!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@
	unknown_6:@$
	unknown_7:@?
	unknown_8:	?%
	unknown_9:??

unknown_10:	?

unknown_11:	?

unknown_12:	?

unknown_13:	?

unknown_14:	?&

unknown_15:??

unknown_16:	?

unknown_17:	?

unknown_18:	?

unknown_19:	?

unknown_20:	?

unknown_21:
??

unknown_22:	?

unknown_23:	?

unknown_24:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_9_layer_call_and_return_conditional_losses_176448o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_5
?
?
M__inference_module_wrapper_95_layer_call_and_return_conditional_losses_176441

args_0:
'dense_19_matmul_readvariableop_resource:	?6
(dense_19_biasadd_readvariableop_resource:
identity??dense_19/BiasAdd/ReadVariableOp?dense_19/MatMul/ReadVariableOp?
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0{
dense_19/MatMulMatMulargs_0&dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_19/BiasAdd/ReadVariableOpReadVariableOp(dense_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_19/BiasAddBiasAdddense_19/MatMul:product:0'dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????h
dense_19/SoftmaxSoftmaxdense_19/BiasAdd:output:0*
T0*'
_output_shapes
:?????????i
IdentityIdentitydense_19/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 2B
dense_19/BiasAdd/ReadVariableOpdense_19/BiasAdd/ReadVariableOp2@
dense_19/MatMul/ReadVariableOpdense_19/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameargs_0
??
?
H__inference_sequential_9_layer_call_and_return_conditional_losses_177534

inputsT
:module_wrapper_85_conv2d_41_conv2d_readvariableop_resource:@I
;module_wrapper_85_conv2d_41_biasadd_readvariableop_resource:@T
:module_wrapper_86_conv2d_42_conv2d_readvariableop_resource:@@I
;module_wrapper_86_conv2d_42_biasadd_readvariableop_resource:@<
.batch_normalization_21_readvariableop_resource:@>
0batch_normalization_21_readvariableop_1_resource:@M
?batch_normalization_21_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_21_fusedbatchnormv3_readvariableop_1_resource:@U
:module_wrapper_88_conv2d_43_conv2d_readvariableop_resource:@?J
;module_wrapper_88_conv2d_43_biasadd_readvariableop_resource:	?V
:module_wrapper_89_conv2d_44_conv2d_readvariableop_resource:??J
;module_wrapper_89_conv2d_44_biasadd_readvariableop_resource:	?=
.batch_normalization_22_readvariableop_resource:	??
0batch_normalization_22_readvariableop_1_resource:	?N
?batch_normalization_22_fusedbatchnormv3_readvariableop_resource:	?P
Abatch_normalization_22_fusedbatchnormv3_readvariableop_1_resource:	?V
:module_wrapper_91_conv2d_45_conv2d_readvariableop_resource:??J
;module_wrapper_91_conv2d_45_biasadd_readvariableop_resource:	?=
.batch_normalization_23_readvariableop_resource:	??
0batch_normalization_23_readvariableop_1_resource:	?N
?batch_normalization_23_fusedbatchnormv3_readvariableop_resource:	?P
Abatch_normalization_23_fusedbatchnormv3_readvariableop_1_resource:	?M
9module_wrapper_94_dense_18_matmul_readvariableop_resource:
??I
:module_wrapper_94_dense_18_biasadd_readvariableop_resource:	?L
9module_wrapper_95_dense_19_matmul_readvariableop_resource:	?H
:module_wrapper_95_dense_19_biasadd_readvariableop_resource:
identity??%batch_normalization_21/AssignNewValue?'batch_normalization_21/AssignNewValue_1?6batch_normalization_21/FusedBatchNormV3/ReadVariableOp?8batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_21/ReadVariableOp?'batch_normalization_21/ReadVariableOp_1?%batch_normalization_22/AssignNewValue?'batch_normalization_22/AssignNewValue_1?6batch_normalization_22/FusedBatchNormV3/ReadVariableOp?8batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_22/ReadVariableOp?'batch_normalization_22/ReadVariableOp_1?%batch_normalization_23/AssignNewValue?'batch_normalization_23/AssignNewValue_1?6batch_normalization_23/FusedBatchNormV3/ReadVariableOp?8batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_23/ReadVariableOp?'batch_normalization_23/ReadVariableOp_1?2module_wrapper_85/conv2d_41/BiasAdd/ReadVariableOp?1module_wrapper_85/conv2d_41/Conv2D/ReadVariableOp?2module_wrapper_86/conv2d_42/BiasAdd/ReadVariableOp?1module_wrapper_86/conv2d_42/Conv2D/ReadVariableOp?2module_wrapper_88/conv2d_43/BiasAdd/ReadVariableOp?1module_wrapper_88/conv2d_43/Conv2D/ReadVariableOp?2module_wrapper_89/conv2d_44/BiasAdd/ReadVariableOp?1module_wrapper_89/conv2d_44/Conv2D/ReadVariableOp?2module_wrapper_91/conv2d_45/BiasAdd/ReadVariableOp?1module_wrapper_91/conv2d_45/Conv2D/ReadVariableOp?1module_wrapper_94/dense_18/BiasAdd/ReadVariableOp?0module_wrapper_94/dense_18/MatMul/ReadVariableOp?1module_wrapper_95/dense_19/BiasAdd/ReadVariableOp?0module_wrapper_95/dense_19/MatMul/ReadVariableOp?
1module_wrapper_85/conv2d_41/Conv2D/ReadVariableOpReadVariableOp:module_wrapper_85_conv2d_41_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
"module_wrapper_85/conv2d_41/Conv2DConv2Dinputs9module_wrapper_85/conv2d_41/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
?
2module_wrapper_85/conv2d_41/BiasAdd/ReadVariableOpReadVariableOp;module_wrapper_85_conv2d_41_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
#module_wrapper_85/conv2d_41/BiasAddBiasAdd+module_wrapper_85/conv2d_41/Conv2D:output:0:module_wrapper_85/conv2d_41/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@?
 module_wrapper_85/conv2d_41/ReluRelu,module_wrapper_85/conv2d_41/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@?
1module_wrapper_86/conv2d_42/Conv2D/ReadVariableOpReadVariableOp:module_wrapper_86_conv2d_42_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
"module_wrapper_86/conv2d_42/Conv2DConv2D.module_wrapper_85/conv2d_41/Relu:activations:09module_wrapper_86/conv2d_42/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
?
2module_wrapper_86/conv2d_42/BiasAdd/ReadVariableOpReadVariableOp;module_wrapper_86_conv2d_42_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
#module_wrapper_86/conv2d_42/BiasAddBiasAdd+module_wrapper_86/conv2d_42/Conv2D:output:0:module_wrapper_86/conv2d_42/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@?
 module_wrapper_86/conv2d_42/ReluRelu,module_wrapper_86/conv2d_42/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@?
*module_wrapper_87/max_pooling2d_26/MaxPoolMaxPool.module_wrapper_86/conv2d_42/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
?
%batch_normalization_21/ReadVariableOpReadVariableOp.batch_normalization_21_readvariableop_resource*
_output_shapes
:@*
dtype0?
'batch_normalization_21/ReadVariableOp_1ReadVariableOp0batch_normalization_21_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
6batch_normalization_21/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_21_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
8batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_21_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
'batch_normalization_21/FusedBatchNormV3FusedBatchNormV33module_wrapper_87/max_pooling2d_26/MaxPool:output:0-batch_normalization_21/ReadVariableOp:value:0/batch_normalization_21/ReadVariableOp_1:value:0>batch_normalization_21/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<?
%batch_normalization_21/AssignNewValueAssignVariableOp?batch_normalization_21_fusedbatchnormv3_readvariableop_resource4batch_normalization_21/FusedBatchNormV3:batch_mean:07^batch_normalization_21/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
'batch_normalization_21/AssignNewValue_1AssignVariableOpAbatch_normalization_21_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_21/FusedBatchNormV3:batch_variance:09^batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(?
1module_wrapper_88/conv2d_43/Conv2D/ReadVariableOpReadVariableOp:module_wrapper_88_conv2d_43_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
"module_wrapper_88/conv2d_43/Conv2DConv2D+batch_normalization_21/FusedBatchNormV3:y:09module_wrapper_88/conv2d_43/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????

?*
paddingVALID*
strides
?
2module_wrapper_88/conv2d_43/BiasAdd/ReadVariableOpReadVariableOp;module_wrapper_88_conv2d_43_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
#module_wrapper_88/conv2d_43/BiasAddBiasAdd+module_wrapper_88/conv2d_43/Conv2D:output:0:module_wrapper_88/conv2d_43/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????

??
 module_wrapper_88/conv2d_43/ReluRelu,module_wrapper_88/conv2d_43/BiasAdd:output:0*
T0*0
_output_shapes
:?????????

??
1module_wrapper_89/conv2d_44/Conv2D/ReadVariableOpReadVariableOp:module_wrapper_89_conv2d_44_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
"module_wrapper_89/conv2d_44/Conv2DConv2D.module_wrapper_88/conv2d_43/Relu:activations:09module_wrapper_89/conv2d_44/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
?
2module_wrapper_89/conv2d_44/BiasAdd/ReadVariableOpReadVariableOp;module_wrapper_89_conv2d_44_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
#module_wrapper_89/conv2d_44/BiasAddBiasAdd+module_wrapper_89/conv2d_44/Conv2D:output:0:module_wrapper_89/conv2d_44/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:???????????
 module_wrapper_89/conv2d_44/ReluRelu,module_wrapper_89/conv2d_44/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
*module_wrapper_90/max_pooling2d_27/MaxPoolMaxPool.module_wrapper_89/conv2d_44/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
?
%batch_normalization_22/ReadVariableOpReadVariableOp.batch_normalization_22_readvariableop_resource*
_output_shapes	
:?*
dtype0?
'batch_normalization_22/ReadVariableOp_1ReadVariableOp0batch_normalization_22_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
6batch_normalization_22/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_22_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
8batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_22_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
'batch_normalization_22/FusedBatchNormV3FusedBatchNormV33module_wrapper_90/max_pooling2d_27/MaxPool:output:0-batch_normalization_22/ReadVariableOp:value:0/batch_normalization_22/ReadVariableOp_1:value:0>batch_normalization_22/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
%batch_normalization_22/AssignNewValueAssignVariableOp?batch_normalization_22_fusedbatchnormv3_readvariableop_resource4batch_normalization_22/FusedBatchNormV3:batch_mean:07^batch_normalization_22/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
'batch_normalization_22/AssignNewValue_1AssignVariableOpAbatch_normalization_22_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_22/FusedBatchNormV3:batch_variance:09^batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(?
1module_wrapper_91/conv2d_45/Conv2D/ReadVariableOpReadVariableOp:module_wrapper_91_conv2d_45_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
"module_wrapper_91/conv2d_45/Conv2DConv2D+batch_normalization_22/FusedBatchNormV3:y:09module_wrapper_91/conv2d_45/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
?
2module_wrapper_91/conv2d_45/BiasAdd/ReadVariableOpReadVariableOp;module_wrapper_91_conv2d_45_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
#module_wrapper_91/conv2d_45/BiasAddBiasAdd+module_wrapper_91/conv2d_45/Conv2D:output:0:module_wrapper_91/conv2d_45/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:???????????
 module_wrapper_91/conv2d_45/ReluRelu,module_wrapper_91/conv2d_45/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
*module_wrapper_92/max_pooling2d_28/MaxPoolMaxPool.module_wrapper_91/conv2d_45/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
?
%batch_normalization_23/ReadVariableOpReadVariableOp.batch_normalization_23_readvariableop_resource*
_output_shapes	
:?*
dtype0?
'batch_normalization_23/ReadVariableOp_1ReadVariableOp0batch_normalization_23_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
6batch_normalization_23/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_23_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
8batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_23_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
'batch_normalization_23/FusedBatchNormV3FusedBatchNormV33module_wrapper_92/max_pooling2d_28/MaxPool:output:0-batch_normalization_23/ReadVariableOp:value:0/batch_normalization_23/ReadVariableOp_1:value:0>batch_normalization_23/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
%batch_normalization_23/AssignNewValueAssignVariableOp?batch_normalization_23_fusedbatchnormv3_readvariableop_resource4batch_normalization_23/FusedBatchNormV3:batch_mean:07^batch_normalization_23/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
'batch_normalization_23/AssignNewValue_1AssignVariableOpAbatch_normalization_23_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_23/FusedBatchNormV3:batch_variance:09^batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(r
!module_wrapper_93/flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ?
#module_wrapper_93/flatten_9/ReshapeReshape+batch_normalization_23/FusedBatchNormV3:y:0*module_wrapper_93/flatten_9/Const:output:0*
T0*(
_output_shapes
:???????????
0module_wrapper_94/dense_18/MatMul/ReadVariableOpReadVariableOp9module_wrapper_94_dense_18_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
!module_wrapper_94/dense_18/MatMulMatMul,module_wrapper_93/flatten_9/Reshape:output:08module_wrapper_94/dense_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
1module_wrapper_94/dense_18/BiasAdd/ReadVariableOpReadVariableOp:module_wrapper_94_dense_18_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
"module_wrapper_94/dense_18/BiasAddBiasAdd+module_wrapper_94/dense_18/MatMul:product:09module_wrapper_94/dense_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
module_wrapper_94/dense_18/ReluRelu+module_wrapper_94/dense_18/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
0module_wrapper_95/dense_19/MatMul/ReadVariableOpReadVariableOp9module_wrapper_95_dense_19_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
!module_wrapper_95/dense_19/MatMulMatMul-module_wrapper_94/dense_18/Relu:activations:08module_wrapper_95/dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
1module_wrapper_95/dense_19/BiasAdd/ReadVariableOpReadVariableOp:module_wrapper_95_dense_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
"module_wrapper_95/dense_19/BiasAddBiasAdd+module_wrapper_95/dense_19/MatMul:product:09module_wrapper_95/dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
"module_wrapper_95/dense_19/SoftmaxSoftmax+module_wrapper_95/dense_19/BiasAdd:output:0*
T0*'
_output_shapes
:?????????{
IdentityIdentity,module_wrapper_95/dense_19/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp&^batch_normalization_21/AssignNewValue(^batch_normalization_21/AssignNewValue_17^batch_normalization_21/FusedBatchNormV3/ReadVariableOp9^batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_21/ReadVariableOp(^batch_normalization_21/ReadVariableOp_1&^batch_normalization_22/AssignNewValue(^batch_normalization_22/AssignNewValue_17^batch_normalization_22/FusedBatchNormV3/ReadVariableOp9^batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_22/ReadVariableOp(^batch_normalization_22/ReadVariableOp_1&^batch_normalization_23/AssignNewValue(^batch_normalization_23/AssignNewValue_17^batch_normalization_23/FusedBatchNormV3/ReadVariableOp9^batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_23/ReadVariableOp(^batch_normalization_23/ReadVariableOp_13^module_wrapper_85/conv2d_41/BiasAdd/ReadVariableOp2^module_wrapper_85/conv2d_41/Conv2D/ReadVariableOp3^module_wrapper_86/conv2d_42/BiasAdd/ReadVariableOp2^module_wrapper_86/conv2d_42/Conv2D/ReadVariableOp3^module_wrapper_88/conv2d_43/BiasAdd/ReadVariableOp2^module_wrapper_88/conv2d_43/Conv2D/ReadVariableOp3^module_wrapper_89/conv2d_44/BiasAdd/ReadVariableOp2^module_wrapper_89/conv2d_44/Conv2D/ReadVariableOp3^module_wrapper_91/conv2d_45/BiasAdd/ReadVariableOp2^module_wrapper_91/conv2d_45/Conv2D/ReadVariableOp2^module_wrapper_94/dense_18/BiasAdd/ReadVariableOp1^module_wrapper_94/dense_18/MatMul/ReadVariableOp2^module_wrapper_95/dense_19/BiasAdd/ReadVariableOp1^module_wrapper_95/dense_19/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : 2N
%batch_normalization_21/AssignNewValue%batch_normalization_21/AssignNewValue2R
'batch_normalization_21/AssignNewValue_1'batch_normalization_21/AssignNewValue_12p
6batch_normalization_21/FusedBatchNormV3/ReadVariableOp6batch_normalization_21/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_21/FusedBatchNormV3/ReadVariableOp_18batch_normalization_21/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_21/ReadVariableOp%batch_normalization_21/ReadVariableOp2R
'batch_normalization_21/ReadVariableOp_1'batch_normalization_21/ReadVariableOp_12N
%batch_normalization_22/AssignNewValue%batch_normalization_22/AssignNewValue2R
'batch_normalization_22/AssignNewValue_1'batch_normalization_22/AssignNewValue_12p
6batch_normalization_22/FusedBatchNormV3/ReadVariableOp6batch_normalization_22/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_22/FusedBatchNormV3/ReadVariableOp_18batch_normalization_22/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_22/ReadVariableOp%batch_normalization_22/ReadVariableOp2R
'batch_normalization_22/ReadVariableOp_1'batch_normalization_22/ReadVariableOp_12N
%batch_normalization_23/AssignNewValue%batch_normalization_23/AssignNewValue2R
'batch_normalization_23/AssignNewValue_1'batch_normalization_23/AssignNewValue_12p
6batch_normalization_23/FusedBatchNormV3/ReadVariableOp6batch_normalization_23/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_23/FusedBatchNormV3/ReadVariableOp_18batch_normalization_23/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_23/ReadVariableOp%batch_normalization_23/ReadVariableOp2R
'batch_normalization_23/ReadVariableOp_1'batch_normalization_23/ReadVariableOp_12h
2module_wrapper_85/conv2d_41/BiasAdd/ReadVariableOp2module_wrapper_85/conv2d_41/BiasAdd/ReadVariableOp2f
1module_wrapper_85/conv2d_41/Conv2D/ReadVariableOp1module_wrapper_85/conv2d_41/Conv2D/ReadVariableOp2h
2module_wrapper_86/conv2d_42/BiasAdd/ReadVariableOp2module_wrapper_86/conv2d_42/BiasAdd/ReadVariableOp2f
1module_wrapper_86/conv2d_42/Conv2D/ReadVariableOp1module_wrapper_86/conv2d_42/Conv2D/ReadVariableOp2h
2module_wrapper_88/conv2d_43/BiasAdd/ReadVariableOp2module_wrapper_88/conv2d_43/BiasAdd/ReadVariableOp2f
1module_wrapper_88/conv2d_43/Conv2D/ReadVariableOp1module_wrapper_88/conv2d_43/Conv2D/ReadVariableOp2h
2module_wrapper_89/conv2d_44/BiasAdd/ReadVariableOp2module_wrapper_89/conv2d_44/BiasAdd/ReadVariableOp2f
1module_wrapper_89/conv2d_44/Conv2D/ReadVariableOp1module_wrapper_89/conv2d_44/Conv2D/ReadVariableOp2h
2module_wrapper_91/conv2d_45/BiasAdd/ReadVariableOp2module_wrapper_91/conv2d_45/BiasAdd/ReadVariableOp2f
1module_wrapper_91/conv2d_45/Conv2D/ReadVariableOp1module_wrapper_91/conv2d_45/Conv2D/ReadVariableOp2f
1module_wrapper_94/dense_18/BiasAdd/ReadVariableOp1module_wrapper_94/dense_18/BiasAdd/ReadVariableOp2d
0module_wrapper_94/dense_18/MatMul/ReadVariableOp0module_wrapper_94/dense_18/MatMul/ReadVariableOp2f
1module_wrapper_95/dense_19/BiasAdd/ReadVariableOp1module_wrapper_95/dense_19/BiasAdd/ReadVariableOp2d
0module_wrapper_95/dense_19/MatMul/ReadVariableOp0module_wrapper_95/dense_19/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
i
M__inference_module_wrapper_93_layer_call_and_return_conditional_losses_178033

args_0
identity`
flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   q
flatten_9/ReshapeReshapeargs_0flatten_9/Const:output:0*
T0*(
_output_shapes
:??????????c
IdentityIdentityflatten_9/Reshape:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameargs_0
?M
?
H__inference_sequential_9_layer_call_and_return_conditional_losses_176448

inputs2
module_wrapper_85_176284:@&
module_wrapper_85_176286:@2
module_wrapper_86_176301:@@&
module_wrapper_86_176303:@+
batch_normalization_21_176313:@+
batch_normalization_21_176315:@+
batch_normalization_21_176317:@+
batch_normalization_21_176319:@3
module_wrapper_88_176334:@?'
module_wrapper_88_176336:	?4
module_wrapper_89_176351:??'
module_wrapper_89_176353:	?,
batch_normalization_22_176363:	?,
batch_normalization_22_176365:	?,
batch_normalization_22_176367:	?,
batch_normalization_22_176369:	?4
module_wrapper_91_176384:??'
module_wrapper_91_176386:	?,
batch_normalization_23_176396:	?,
batch_normalization_23_176398:	?,
batch_normalization_23_176400:	?,
batch_normalization_23_176402:	?,
module_wrapper_94_176425:
??'
module_wrapper_94_176427:	?+
module_wrapper_95_176442:	?&
module_wrapper_95_176444:
identity??.batch_normalization_21/StatefulPartitionedCall?.batch_normalization_22/StatefulPartitionedCall?.batch_normalization_23/StatefulPartitionedCall?)module_wrapper_85/StatefulPartitionedCall?)module_wrapper_86/StatefulPartitionedCall?)module_wrapper_88/StatefulPartitionedCall?)module_wrapper_89/StatefulPartitionedCall?)module_wrapper_91/StatefulPartitionedCall?)module_wrapper_94/StatefulPartitionedCall?)module_wrapper_95/StatefulPartitionedCall?
)module_wrapper_85/StatefulPartitionedCallStatefulPartitionedCallinputsmodule_wrapper_85_176284module_wrapper_85_176286*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_85_layer_call_and_return_conditional_losses_176283?
)module_wrapper_86/StatefulPartitionedCallStatefulPartitionedCall2module_wrapper_85/StatefulPartitionedCall:output:0module_wrapper_86_176301module_wrapper_86_176303*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_86_layer_call_and_return_conditional_losses_176300?
!module_wrapper_87/PartitionedCallPartitionedCall2module_wrapper_86/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_87_layer_call_and_return_conditional_losses_176311?
.batch_normalization_21/StatefulPartitionedCallStatefulPartitionedCall*module_wrapper_87/PartitionedCall:output:0batch_normalization_21_176313batch_normalization_21_176315batch_normalization_21_176317batch_normalization_21_176319*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_176095?
)module_wrapper_88/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_21/StatefulPartitionedCall:output:0module_wrapper_88_176334module_wrapper_88_176336*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????

?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_88_layer_call_and_return_conditional_losses_176333?
)module_wrapper_89/StatefulPartitionedCallStatefulPartitionedCall2module_wrapper_88/StatefulPartitionedCall:output:0module_wrapper_89_176351module_wrapper_89_176353*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_89_layer_call_and_return_conditional_losses_176350?
!module_wrapper_90/PartitionedCallPartitionedCall2module_wrapper_89/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_90_layer_call_and_return_conditional_losses_176361?
.batch_normalization_22/StatefulPartitionedCallStatefulPartitionedCall*module_wrapper_90/PartitionedCall:output:0batch_normalization_22_176363batch_normalization_22_176365batch_normalization_22_176367batch_normalization_22_176369*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_176159?
)module_wrapper_91/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_22/StatefulPartitionedCall:output:0module_wrapper_91_176384module_wrapper_91_176386*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_91_layer_call_and_return_conditional_losses_176383?
!module_wrapper_92/PartitionedCallPartitionedCall2module_wrapper_91/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_92_layer_call_and_return_conditional_losses_176394?
.batch_normalization_23/StatefulPartitionedCallStatefulPartitionedCall*module_wrapper_92/PartitionedCall:output:0batch_normalization_23_176396batch_normalization_23_176398batch_normalization_23_176400batch_normalization_23_176402*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_176223?
!module_wrapper_93/PartitionedCallPartitionedCall7batch_normalization_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_93_layer_call_and_return_conditional_losses_176411?
)module_wrapper_94/StatefulPartitionedCallStatefulPartitionedCall*module_wrapper_93/PartitionedCall:output:0module_wrapper_94_176425module_wrapper_94_176427*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_94_layer_call_and_return_conditional_losses_176424?
)module_wrapper_95/StatefulPartitionedCallStatefulPartitionedCall2module_wrapper_94/StatefulPartitionedCall:output:0module_wrapper_95_176442module_wrapper_95_176444*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_95_layer_call_and_return_conditional_losses_176441?
IdentityIdentity2module_wrapper_95/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp/^batch_normalization_21/StatefulPartitionedCall/^batch_normalization_22/StatefulPartitionedCall/^batch_normalization_23/StatefulPartitionedCall*^module_wrapper_85/StatefulPartitionedCall*^module_wrapper_86/StatefulPartitionedCall*^module_wrapper_88/StatefulPartitionedCall*^module_wrapper_89/StatefulPartitionedCall*^module_wrapper_91/StatefulPartitionedCall*^module_wrapper_94/StatefulPartitionedCall*^module_wrapper_95/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_21/StatefulPartitionedCall.batch_normalization_21/StatefulPartitionedCall2`
.batch_normalization_22/StatefulPartitionedCall.batch_normalization_22/StatefulPartitionedCall2`
.batch_normalization_23/StatefulPartitionedCall.batch_normalization_23/StatefulPartitionedCall2V
)module_wrapper_85/StatefulPartitionedCall)module_wrapper_85/StatefulPartitionedCall2V
)module_wrapper_86/StatefulPartitionedCall)module_wrapper_86/StatefulPartitionedCall2V
)module_wrapper_88/StatefulPartitionedCall)module_wrapper_88/StatefulPartitionedCall2V
)module_wrapper_89/StatefulPartitionedCall)module_wrapper_89/StatefulPartitionedCall2V
)module_wrapper_91/StatefulPartitionedCall)module_wrapper_91/StatefulPartitionedCall2V
)module_wrapper_94/StatefulPartitionedCall)module_wrapper_94/StatefulPartitionedCall2V
)module_wrapper_95/StatefulPartitionedCall)module_wrapper_95/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
2__inference_module_wrapper_88_layer_call_fn_177718

args_0"
unknown:@?
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????

?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_88_layer_call_and_return_conditional_losses_176333x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????

?`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameargs_0
?
?
M__inference_module_wrapper_88_layer_call_and_return_conditional_losses_177738

args_0C
(conv2d_43_conv2d_readvariableop_resource:@?8
)conv2d_43_biasadd_readvariableop_resource:	?
identity?? conv2d_43/BiasAdd/ReadVariableOp?conv2d_43/Conv2D/ReadVariableOp?
conv2d_43/Conv2D/ReadVariableOpReadVariableOp(conv2d_43_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
conv2d_43/Conv2DConv2Dargs_0'conv2d_43/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????

?*
paddingVALID*
strides
?
 conv2d_43/BiasAdd/ReadVariableOpReadVariableOp)conv2d_43_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
conv2d_43/BiasAddBiasAddconv2d_43/Conv2D:output:0(conv2d_43/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????

?m
conv2d_43/ReluReluconv2d_43/BiasAdd:output:0*
T0*0
_output_shapes
:?????????

?t
IdentityIdentityconv2d_43/Relu:activations:0^NoOp*
T0*0
_output_shapes
:?????????

??
NoOpNoOp!^conv2d_43/BiasAdd/ReadVariableOp ^conv2d_43/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 2D
 conv2d_43/BiasAdd/ReadVariableOp conv2d_43/BiasAdd/ReadVariableOp2B
conv2d_43/Conv2D/ReadVariableOpconv2d_43/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameargs_0
?
?
M__inference_module_wrapper_94_layer_call_and_return_conditional_losses_176556

args_0;
'dense_18_matmul_readvariableop_resource:
??7
(dense_18_biasadd_readvariableop_resource:	?
identity??dense_18/BiasAdd/ReadVariableOp?dense_18/MatMul/ReadVariableOp?
dense_18/MatMul/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0|
dense_18/MatMulMatMulargs_0&dense_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_18/BiasAdd/ReadVariableOpReadVariableOp(dense_18_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_18/BiasAddBiasAdddense_18/MatMul:product:0'dense_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????c
dense_18/ReluReludense_18/BiasAdd:output:0*
T0*(
_output_shapes
:??????????k
IdentityIdentitydense_18/Relu:activations:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 2B
dense_18/BiasAdd/ReadVariableOpdense_18/BiasAdd/ReadVariableOp2@
dense_18/MatMul/ReadVariableOpdense_18/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameargs_0
?
?
M__inference_module_wrapper_88_layer_call_and_return_conditional_losses_176333

args_0C
(conv2d_43_conv2d_readvariableop_resource:@?8
)conv2d_43_biasadd_readvariableop_resource:	?
identity?? conv2d_43/BiasAdd/ReadVariableOp?conv2d_43/Conv2D/ReadVariableOp?
conv2d_43/Conv2D/ReadVariableOpReadVariableOp(conv2d_43_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
conv2d_43/Conv2DConv2Dargs_0'conv2d_43/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????

?*
paddingVALID*
strides
?
 conv2d_43/BiasAdd/ReadVariableOpReadVariableOp)conv2d_43_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
conv2d_43/BiasAddBiasAddconv2d_43/Conv2D:output:0(conv2d_43/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????

?m
conv2d_43/ReluReluconv2d_43/BiasAdd:output:0*
T0*0
_output_shapes
:?????????

?t
IdentityIdentityconv2d_43/Relu:activations:0^NoOp*
T0*0
_output_shapes
:?????????

??
NoOpNoOp!^conv2d_43/BiasAdd/ReadVariableOp ^conv2d_43/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 2D
 conv2d_43/BiasAdd/ReadVariableOp conv2d_43/BiasAdd/ReadVariableOp2B
conv2d_43/Conv2D/ReadVariableOpconv2d_43/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameargs_0
?
?
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_176126

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
M__inference_module_wrapper_86_layer_call_and_return_conditional_losses_177604

args_0B
(conv2d_42_conv2d_readvariableop_resource:@@7
)conv2d_42_biasadd_readvariableop_resource:@
identity?? conv2d_42/BiasAdd/ReadVariableOp?conv2d_42/Conv2D/ReadVariableOp?
conv2d_42/Conv2D/ReadVariableOpReadVariableOp(conv2d_42_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
conv2d_42/Conv2DConv2Dargs_0'conv2d_42/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
?
 conv2d_42/BiasAdd/ReadVariableOpReadVariableOp)conv2d_42_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_42/BiasAddBiasAddconv2d_42/Conv2D:output:0(conv2d_42/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@l
conv2d_42/ReluReluconv2d_42/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@s
IdentityIdentityconv2d_42/Relu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@?
NoOpNoOp!^conv2d_42/BiasAdd/ReadVariableOp ^conv2d_42/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 2D
 conv2d_42/BiasAdd/ReadVariableOp conv2d_42/BiasAdd/ReadVariableOp2B
conv2d_42/Conv2D/ReadVariableOpconv2d_42/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameargs_0
?M
?
H__inference_sequential_9_layer_call_and_return_conditional_losses_176907

inputs2
module_wrapper_85_176840:@&
module_wrapper_85_176842:@2
module_wrapper_86_176845:@@&
module_wrapper_86_176847:@+
batch_normalization_21_176851:@+
batch_normalization_21_176853:@+
batch_normalization_21_176855:@+
batch_normalization_21_176857:@3
module_wrapper_88_176860:@?'
module_wrapper_88_176862:	?4
module_wrapper_89_176865:??'
module_wrapper_89_176867:	?,
batch_normalization_22_176871:	?,
batch_normalization_22_176873:	?,
batch_normalization_22_176875:	?,
batch_normalization_22_176877:	?4
module_wrapper_91_176880:??'
module_wrapper_91_176882:	?,
batch_normalization_23_176886:	?,
batch_normalization_23_176888:	?,
batch_normalization_23_176890:	?,
batch_normalization_23_176892:	?,
module_wrapper_94_176896:
??'
module_wrapper_94_176898:	?+
module_wrapper_95_176901:	?&
module_wrapper_95_176903:
identity??.batch_normalization_21/StatefulPartitionedCall?.batch_normalization_22/StatefulPartitionedCall?.batch_normalization_23/StatefulPartitionedCall?)module_wrapper_85/StatefulPartitionedCall?)module_wrapper_86/StatefulPartitionedCall?)module_wrapper_88/StatefulPartitionedCall?)module_wrapper_89/StatefulPartitionedCall?)module_wrapper_91/StatefulPartitionedCall?)module_wrapper_94/StatefulPartitionedCall?)module_wrapper_95/StatefulPartitionedCall?
)module_wrapper_85/StatefulPartitionedCallStatefulPartitionedCallinputsmodule_wrapper_85_176840module_wrapper_85_176842*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_85_layer_call_and_return_conditional_losses_176771?
)module_wrapper_86/StatefulPartitionedCallStatefulPartitionedCall2module_wrapper_85/StatefulPartitionedCall:output:0module_wrapper_86_176845module_wrapper_86_176847*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_86_layer_call_and_return_conditional_losses_176741?
!module_wrapper_87/PartitionedCallPartitionedCall2module_wrapper_86/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_87_layer_call_and_return_conditional_losses_176715?
.batch_normalization_21/StatefulPartitionedCallStatefulPartitionedCall*module_wrapper_87/PartitionedCall:output:0batch_normalization_21_176851batch_normalization_21_176853batch_normalization_21_176855batch_normalization_21_176857*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_176126?
)module_wrapper_88/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_21/StatefulPartitionedCall:output:0module_wrapper_88_176860module_wrapper_88_176862*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????

?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_88_layer_call_and_return_conditional_losses_176695?
)module_wrapper_89/StatefulPartitionedCallStatefulPartitionedCall2module_wrapper_88/StatefulPartitionedCall:output:0module_wrapper_89_176865module_wrapper_89_176867*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_89_layer_call_and_return_conditional_losses_176665?
!module_wrapper_90/PartitionedCallPartitionedCall2module_wrapper_89/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_90_layer_call_and_return_conditional_losses_176639?
.batch_normalization_22/StatefulPartitionedCallStatefulPartitionedCall*module_wrapper_90/PartitionedCall:output:0batch_normalization_22_176871batch_normalization_22_176873batch_normalization_22_176875batch_normalization_22_176877*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_176190?
)module_wrapper_91/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_22/StatefulPartitionedCall:output:0module_wrapper_91_176880module_wrapper_91_176882*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_91_layer_call_and_return_conditional_losses_176619?
!module_wrapper_92/PartitionedCallPartitionedCall2module_wrapper_91/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_92_layer_call_and_return_conditional_losses_176593?
.batch_normalization_23/StatefulPartitionedCallStatefulPartitionedCall*module_wrapper_92/PartitionedCall:output:0batch_normalization_23_176886batch_normalization_23_176888batch_normalization_23_176890batch_normalization_23_176892*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_176254?
!module_wrapper_93/PartitionedCallPartitionedCall7batch_normalization_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_93_layer_call_and_return_conditional_losses_176577?
)module_wrapper_94/StatefulPartitionedCallStatefulPartitionedCall*module_wrapper_93/PartitionedCall:output:0module_wrapper_94_176896module_wrapper_94_176898*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_94_layer_call_and_return_conditional_losses_176556?
)module_wrapper_95/StatefulPartitionedCallStatefulPartitionedCall2module_wrapper_94/StatefulPartitionedCall:output:0module_wrapper_95_176901module_wrapper_95_176903*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_95_layer_call_and_return_conditional_losses_176526?
IdentityIdentity2module_wrapper_95/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp/^batch_normalization_21/StatefulPartitionedCall/^batch_normalization_22/StatefulPartitionedCall/^batch_normalization_23/StatefulPartitionedCall*^module_wrapper_85/StatefulPartitionedCall*^module_wrapper_86/StatefulPartitionedCall*^module_wrapper_88/StatefulPartitionedCall*^module_wrapper_89/StatefulPartitionedCall*^module_wrapper_91/StatefulPartitionedCall*^module_wrapper_94/StatefulPartitionedCall*^module_wrapper_95/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_21/StatefulPartitionedCall.batch_normalization_21/StatefulPartitionedCall2`
.batch_normalization_22/StatefulPartitionedCall.batch_normalization_22/StatefulPartitionedCall2`
.batch_normalization_23/StatefulPartitionedCall.batch_normalization_23/StatefulPartitionedCall2V
)module_wrapper_85/StatefulPartitionedCall)module_wrapper_85/StatefulPartitionedCall2V
)module_wrapper_86/StatefulPartitionedCall)module_wrapper_86/StatefulPartitionedCall2V
)module_wrapper_88/StatefulPartitionedCall)module_wrapper_88/StatefulPartitionedCall2V
)module_wrapper_89/StatefulPartitionedCall)module_wrapper_89/StatefulPartitionedCall2V
)module_wrapper_91/StatefulPartitionedCall)module_wrapper_91/StatefulPartitionedCall2V
)module_wrapper_94/StatefulPartitionedCall)module_wrapper_94/StatefulPartitionedCall2V
)module_wrapper_95/StatefulPartitionedCall)module_wrapper_95/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
M
1__inference_max_pooling2d_27_layer_call_fn_177821

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_27_layer_call_and_return_conditional_losses_177815?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
i
M__inference_module_wrapper_92_layer_call_and_return_conditional_losses_177943

args_0
identity?
max_pooling2d_28/MaxPoolMaxPoolargs_0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
r
IdentityIdentity!max_pooling2d_28/MaxPool:output:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameargs_0
ɻ
?5
"__inference__traced_restore_178584
file_prefix;
-assignvariableop_batch_normalization_21_gamma:@<
.assignvariableop_1_batch_normalization_21_beta:@C
5assignvariableop_2_batch_normalization_21_moving_mean:@G
9assignvariableop_3_batch_normalization_21_moving_variance:@>
/assignvariableop_4_batch_normalization_22_gamma:	?=
.assignvariableop_5_batch_normalization_22_beta:	?D
5assignvariableop_6_batch_normalization_22_moving_mean:	?H
9assignvariableop_7_batch_normalization_22_moving_variance:	?>
/assignvariableop_8_batch_normalization_23_gamma:	?=
.assignvariableop_9_batch_normalization_23_beta:	?E
6assignvariableop_10_batch_normalization_23_moving_mean:	?I
:assignvariableop_11_batch_normalization_23_moving_variance:	?P
6assignvariableop_12_module_wrapper_85_conv2d_41_kernel:@B
4assignvariableop_13_module_wrapper_85_conv2d_41_bias:@P
6assignvariableop_14_module_wrapper_86_conv2d_42_kernel:@@B
4assignvariableop_15_module_wrapper_86_conv2d_42_bias:@Q
6assignvariableop_16_module_wrapper_88_conv2d_43_kernel:@?C
4assignvariableop_17_module_wrapper_88_conv2d_43_bias:	?R
6assignvariableop_18_module_wrapper_89_conv2d_44_kernel:??C
4assignvariableop_19_module_wrapper_89_conv2d_44_bias:	?R
6assignvariableop_20_module_wrapper_91_conv2d_45_kernel:??C
4assignvariableop_21_module_wrapper_91_conv2d_45_bias:	?I
5assignvariableop_22_module_wrapper_94_dense_18_kernel:
??B
3assignvariableop_23_module_wrapper_94_dense_18_bias:	?H
5assignvariableop_24_module_wrapper_95_dense_19_kernel:	?A
3assignvariableop_25_module_wrapper_95_dense_19_bias:'
assignvariableop_26_iteration:	 +
!assignvariableop_27_learning_rate: W
=assignvariableop_28_adam_m_module_wrapper_85_conv2d_41_kernel:@W
=assignvariableop_29_adam_v_module_wrapper_85_conv2d_41_kernel:@I
;assignvariableop_30_adam_m_module_wrapper_85_conv2d_41_bias:@I
;assignvariableop_31_adam_v_module_wrapper_85_conv2d_41_bias:@W
=assignvariableop_32_adam_m_module_wrapper_86_conv2d_42_kernel:@@W
=assignvariableop_33_adam_v_module_wrapper_86_conv2d_42_kernel:@@I
;assignvariableop_34_adam_m_module_wrapper_86_conv2d_42_bias:@I
;assignvariableop_35_adam_v_module_wrapper_86_conv2d_42_bias:@E
7assignvariableop_36_adam_m_batch_normalization_21_gamma:@E
7assignvariableop_37_adam_v_batch_normalization_21_gamma:@D
6assignvariableop_38_adam_m_batch_normalization_21_beta:@D
6assignvariableop_39_adam_v_batch_normalization_21_beta:@X
=assignvariableop_40_adam_m_module_wrapper_88_conv2d_43_kernel:@?X
=assignvariableop_41_adam_v_module_wrapper_88_conv2d_43_kernel:@?J
;assignvariableop_42_adam_m_module_wrapper_88_conv2d_43_bias:	?J
;assignvariableop_43_adam_v_module_wrapper_88_conv2d_43_bias:	?Y
=assignvariableop_44_adam_m_module_wrapper_89_conv2d_44_kernel:??Y
=assignvariableop_45_adam_v_module_wrapper_89_conv2d_44_kernel:??J
;assignvariableop_46_adam_m_module_wrapper_89_conv2d_44_bias:	?J
;assignvariableop_47_adam_v_module_wrapper_89_conv2d_44_bias:	?F
7assignvariableop_48_adam_m_batch_normalization_22_gamma:	?F
7assignvariableop_49_adam_v_batch_normalization_22_gamma:	?E
6assignvariableop_50_adam_m_batch_normalization_22_beta:	?E
6assignvariableop_51_adam_v_batch_normalization_22_beta:	?Y
=assignvariableop_52_adam_m_module_wrapper_91_conv2d_45_kernel:??Y
=assignvariableop_53_adam_v_module_wrapper_91_conv2d_45_kernel:??J
;assignvariableop_54_adam_m_module_wrapper_91_conv2d_45_bias:	?J
;assignvariableop_55_adam_v_module_wrapper_91_conv2d_45_bias:	?F
7assignvariableop_56_adam_m_batch_normalization_23_gamma:	?F
7assignvariableop_57_adam_v_batch_normalization_23_gamma:	?E
6assignvariableop_58_adam_m_batch_normalization_23_beta:	?E
6assignvariableop_59_adam_v_batch_normalization_23_beta:	?P
<assignvariableop_60_adam_m_module_wrapper_94_dense_18_kernel:
??P
<assignvariableop_61_adam_v_module_wrapper_94_dense_18_kernel:
??I
:assignvariableop_62_adam_m_module_wrapper_94_dense_18_bias:	?I
:assignvariableop_63_adam_v_module_wrapper_94_dense_18_bias:	?O
<assignvariableop_64_adam_m_module_wrapper_95_dense_19_kernel:	?O
<assignvariableop_65_adam_v_module_wrapper_95_dense_19_kernel:	?H
:assignvariableop_66_adam_m_module_wrapper_95_dense_19_bias:H
:assignvariableop_67_adam_v_module_wrapper_95_dense_19_bias:%
assignvariableop_68_total_1: %
assignvariableop_69_count_1: #
assignvariableop_70_total: #
assignvariableop_71_count: 
identity_73??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:I*
dtype0*?
value?B?IB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:I*
dtype0*?
value?B?IB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*W
dtypesM
K2I	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp-assignvariableop_batch_normalization_21_gammaIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp.assignvariableop_1_batch_normalization_21_betaIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp5assignvariableop_2_batch_normalization_21_moving_meanIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp9assignvariableop_3_batch_normalization_21_moving_varianceIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp/assignvariableop_4_batch_normalization_22_gammaIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp.assignvariableop_5_batch_normalization_22_betaIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp5assignvariableop_6_batch_normalization_22_moving_meanIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp9assignvariableop_7_batch_normalization_22_moving_varianceIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp/assignvariableop_8_batch_normalization_23_gammaIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp.assignvariableop_9_batch_normalization_23_betaIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp6assignvariableop_10_batch_normalization_23_moving_meanIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp:assignvariableop_11_batch_normalization_23_moving_varianceIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp6assignvariableop_12_module_wrapper_85_conv2d_41_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp4assignvariableop_13_module_wrapper_85_conv2d_41_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp6assignvariableop_14_module_wrapper_86_conv2d_42_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp4assignvariableop_15_module_wrapper_86_conv2d_42_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp6assignvariableop_16_module_wrapper_88_conv2d_43_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp4assignvariableop_17_module_wrapper_88_conv2d_43_biasIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp6assignvariableop_18_module_wrapper_89_conv2d_44_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp4assignvariableop_19_module_wrapper_89_conv2d_44_biasIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp6assignvariableop_20_module_wrapper_91_conv2d_45_kernelIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp4assignvariableop_21_module_wrapper_91_conv2d_45_biasIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp5assignvariableop_22_module_wrapper_94_dense_18_kernelIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp3assignvariableop_23_module_wrapper_94_dense_18_biasIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp5assignvariableop_24_module_wrapper_95_dense_19_kernelIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp3assignvariableop_25_module_wrapper_95_dense_19_biasIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_26AssignVariableOpassignvariableop_26_iterationIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp!assignvariableop_27_learning_rateIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp=assignvariableop_28_adam_m_module_wrapper_85_conv2d_41_kernelIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp=assignvariableop_29_adam_v_module_wrapper_85_conv2d_41_kernelIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp;assignvariableop_30_adam_m_module_wrapper_85_conv2d_41_biasIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOp;assignvariableop_31_adam_v_module_wrapper_85_conv2d_41_biasIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOp=assignvariableop_32_adam_m_module_wrapper_86_conv2d_42_kernelIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOp=assignvariableop_33_adam_v_module_wrapper_86_conv2d_42_kernelIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOp;assignvariableop_34_adam_m_module_wrapper_86_conv2d_42_biasIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_35AssignVariableOp;assignvariableop_35_adam_v_module_wrapper_86_conv2d_42_biasIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOp7assignvariableop_36_adam_m_batch_normalization_21_gammaIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_37AssignVariableOp7assignvariableop_37_adam_v_batch_normalization_21_gammaIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_38AssignVariableOp6assignvariableop_38_adam_m_batch_normalization_21_betaIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_39AssignVariableOp6assignvariableop_39_adam_v_batch_normalization_21_betaIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_40AssignVariableOp=assignvariableop_40_adam_m_module_wrapper_88_conv2d_43_kernelIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_41AssignVariableOp=assignvariableop_41_adam_v_module_wrapper_88_conv2d_43_kernelIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_42AssignVariableOp;assignvariableop_42_adam_m_module_wrapper_88_conv2d_43_biasIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_43AssignVariableOp;assignvariableop_43_adam_v_module_wrapper_88_conv2d_43_biasIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_44AssignVariableOp=assignvariableop_44_adam_m_module_wrapper_89_conv2d_44_kernelIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_45AssignVariableOp=assignvariableop_45_adam_v_module_wrapper_89_conv2d_44_kernelIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_46AssignVariableOp;assignvariableop_46_adam_m_module_wrapper_89_conv2d_44_biasIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_47AssignVariableOp;assignvariableop_47_adam_v_module_wrapper_89_conv2d_44_biasIdentity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_48AssignVariableOp7assignvariableop_48_adam_m_batch_normalization_22_gammaIdentity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_49AssignVariableOp7assignvariableop_49_adam_v_batch_normalization_22_gammaIdentity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_50AssignVariableOp6assignvariableop_50_adam_m_batch_normalization_22_betaIdentity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_51AssignVariableOp6assignvariableop_51_adam_v_batch_normalization_22_betaIdentity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_52AssignVariableOp=assignvariableop_52_adam_m_module_wrapper_91_conv2d_45_kernelIdentity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_53AssignVariableOp=assignvariableop_53_adam_v_module_wrapper_91_conv2d_45_kernelIdentity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_54AssignVariableOp;assignvariableop_54_adam_m_module_wrapper_91_conv2d_45_biasIdentity_54:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_55AssignVariableOp;assignvariableop_55_adam_v_module_wrapper_91_conv2d_45_biasIdentity_55:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_56AssignVariableOp7assignvariableop_56_adam_m_batch_normalization_23_gammaIdentity_56:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_57AssignVariableOp7assignvariableop_57_adam_v_batch_normalization_23_gammaIdentity_57:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_58AssignVariableOp6assignvariableop_58_adam_m_batch_normalization_23_betaIdentity_58:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_59AssignVariableOp6assignvariableop_59_adam_v_batch_normalization_23_betaIdentity_59:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_60AssignVariableOp<assignvariableop_60_adam_m_module_wrapper_94_dense_18_kernelIdentity_60:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_61AssignVariableOp<assignvariableop_61_adam_v_module_wrapper_94_dense_18_kernelIdentity_61:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_62AssignVariableOp:assignvariableop_62_adam_m_module_wrapper_94_dense_18_biasIdentity_62:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_63AssignVariableOp:assignvariableop_63_adam_v_module_wrapper_94_dense_18_biasIdentity_63:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_64AssignVariableOp<assignvariableop_64_adam_m_module_wrapper_95_dense_19_kernelIdentity_64:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_65AssignVariableOp<assignvariableop_65_adam_v_module_wrapper_95_dense_19_kernelIdentity_65:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_66AssignVariableOp:assignvariableop_66_adam_m_module_wrapper_95_dense_19_biasIdentity_66:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_67AssignVariableOp:assignvariableop_67_adam_v_module_wrapper_95_dense_19_biasIdentity_67:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_68AssignVariableOpassignvariableop_68_total_1Identity_68:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_69AssignVariableOpassignvariableop_69_count_1Identity_69:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_70AssignVariableOpassignvariableop_70_totalIdentity_70:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_71AssignVariableOpassignvariableop_71_countIdentity_71:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 ?
Identity_72Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_73IdentityIdentity_72:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_73Identity_73:output:0*?
_input_shapes?
?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_176095

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
M__inference_module_wrapper_89_layer_call_and_return_conditional_losses_176350

args_0D
(conv2d_44_conv2d_readvariableop_resource:??8
)conv2d_44_biasadd_readvariableop_resource:	?
identity?? conv2d_44/BiasAdd/ReadVariableOp?conv2d_44/Conv2D/ReadVariableOp?
conv2d_44/Conv2D/ReadVariableOpReadVariableOp(conv2d_44_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
conv2d_44/Conv2DConv2Dargs_0'conv2d_44/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
?
 conv2d_44/BiasAdd/ReadVariableOpReadVariableOp)conv2d_44_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
conv2d_44/BiasAddBiasAddconv2d_44/Conv2D:output:0(conv2d_44/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????m
conv2d_44/ReluReluconv2d_44/BiasAdd:output:0*
T0*0
_output_shapes
:??????????t
IdentityIdentityconv2d_44/Relu:activations:0^NoOp*
T0*0
_output_shapes
:???????????
NoOpNoOp!^conv2d_44/BiasAdd/ReadVariableOp ^conv2d_44/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????

?: : 2D
 conv2d_44/BiasAdd/ReadVariableOp conv2d_44/BiasAdd/ReadVariableOp2B
conv2d_44/Conv2D/ReadVariableOpconv2d_44/Conv2D/ReadVariableOp:X T
0
_output_shapes
:?????????

?
 
_user_specified_nameargs_0
?
?
$__inference_signature_wrapper_177220
input_5!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@
	unknown_6:@$
	unknown_7:@?
	unknown_8:	?%
	unknown_9:??

unknown_10:	?

unknown_11:	?

unknown_12:	?

unknown_13:	?

unknown_14:	?&

unknown_15:??

unknown_16:	?

unknown_17:	?

unknown_18:	?

unknown_19:	?

unknown_20:	?

unknown_21:
??

unknown_22:	?

unknown_23:	?

unknown_24:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__wrapped_model_176073o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_5
??
?
!__inference__wrapped_model_176073
input_5a
Gsequential_9_module_wrapper_85_conv2d_41_conv2d_readvariableop_resource:@V
Hsequential_9_module_wrapper_85_conv2d_41_biasadd_readvariableop_resource:@a
Gsequential_9_module_wrapper_86_conv2d_42_conv2d_readvariableop_resource:@@V
Hsequential_9_module_wrapper_86_conv2d_42_biasadd_readvariableop_resource:@I
;sequential_9_batch_normalization_21_readvariableop_resource:@K
=sequential_9_batch_normalization_21_readvariableop_1_resource:@Z
Lsequential_9_batch_normalization_21_fusedbatchnormv3_readvariableop_resource:@\
Nsequential_9_batch_normalization_21_fusedbatchnormv3_readvariableop_1_resource:@b
Gsequential_9_module_wrapper_88_conv2d_43_conv2d_readvariableop_resource:@?W
Hsequential_9_module_wrapper_88_conv2d_43_biasadd_readvariableop_resource:	?c
Gsequential_9_module_wrapper_89_conv2d_44_conv2d_readvariableop_resource:??W
Hsequential_9_module_wrapper_89_conv2d_44_biasadd_readvariableop_resource:	?J
;sequential_9_batch_normalization_22_readvariableop_resource:	?L
=sequential_9_batch_normalization_22_readvariableop_1_resource:	?[
Lsequential_9_batch_normalization_22_fusedbatchnormv3_readvariableop_resource:	?]
Nsequential_9_batch_normalization_22_fusedbatchnormv3_readvariableop_1_resource:	?c
Gsequential_9_module_wrapper_91_conv2d_45_conv2d_readvariableop_resource:??W
Hsequential_9_module_wrapper_91_conv2d_45_biasadd_readvariableop_resource:	?J
;sequential_9_batch_normalization_23_readvariableop_resource:	?L
=sequential_9_batch_normalization_23_readvariableop_1_resource:	?[
Lsequential_9_batch_normalization_23_fusedbatchnormv3_readvariableop_resource:	?]
Nsequential_9_batch_normalization_23_fusedbatchnormv3_readvariableop_1_resource:	?Z
Fsequential_9_module_wrapper_94_dense_18_matmul_readvariableop_resource:
??V
Gsequential_9_module_wrapper_94_dense_18_biasadd_readvariableop_resource:	?Y
Fsequential_9_module_wrapper_95_dense_19_matmul_readvariableop_resource:	?U
Gsequential_9_module_wrapper_95_dense_19_biasadd_readvariableop_resource:
identity??Csequential_9/batch_normalization_21/FusedBatchNormV3/ReadVariableOp?Esequential_9/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1?2sequential_9/batch_normalization_21/ReadVariableOp?4sequential_9/batch_normalization_21/ReadVariableOp_1?Csequential_9/batch_normalization_22/FusedBatchNormV3/ReadVariableOp?Esequential_9/batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1?2sequential_9/batch_normalization_22/ReadVariableOp?4sequential_9/batch_normalization_22/ReadVariableOp_1?Csequential_9/batch_normalization_23/FusedBatchNormV3/ReadVariableOp?Esequential_9/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1?2sequential_9/batch_normalization_23/ReadVariableOp?4sequential_9/batch_normalization_23/ReadVariableOp_1??sequential_9/module_wrapper_85/conv2d_41/BiasAdd/ReadVariableOp?>sequential_9/module_wrapper_85/conv2d_41/Conv2D/ReadVariableOp??sequential_9/module_wrapper_86/conv2d_42/BiasAdd/ReadVariableOp?>sequential_9/module_wrapper_86/conv2d_42/Conv2D/ReadVariableOp??sequential_9/module_wrapper_88/conv2d_43/BiasAdd/ReadVariableOp?>sequential_9/module_wrapper_88/conv2d_43/Conv2D/ReadVariableOp??sequential_9/module_wrapper_89/conv2d_44/BiasAdd/ReadVariableOp?>sequential_9/module_wrapper_89/conv2d_44/Conv2D/ReadVariableOp??sequential_9/module_wrapper_91/conv2d_45/BiasAdd/ReadVariableOp?>sequential_9/module_wrapper_91/conv2d_45/Conv2D/ReadVariableOp?>sequential_9/module_wrapper_94/dense_18/BiasAdd/ReadVariableOp?=sequential_9/module_wrapper_94/dense_18/MatMul/ReadVariableOp?>sequential_9/module_wrapper_95/dense_19/BiasAdd/ReadVariableOp?=sequential_9/module_wrapper_95/dense_19/MatMul/ReadVariableOp?
>sequential_9/module_wrapper_85/conv2d_41/Conv2D/ReadVariableOpReadVariableOpGsequential_9_module_wrapper_85_conv2d_41_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
/sequential_9/module_wrapper_85/conv2d_41/Conv2DConv2Dinput_5Fsequential_9/module_wrapper_85/conv2d_41/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
?
?sequential_9/module_wrapper_85/conv2d_41/BiasAdd/ReadVariableOpReadVariableOpHsequential_9_module_wrapper_85_conv2d_41_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
0sequential_9/module_wrapper_85/conv2d_41/BiasAddBiasAdd8sequential_9/module_wrapper_85/conv2d_41/Conv2D:output:0Gsequential_9/module_wrapper_85/conv2d_41/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@?
-sequential_9/module_wrapper_85/conv2d_41/ReluRelu9sequential_9/module_wrapper_85/conv2d_41/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@?
>sequential_9/module_wrapper_86/conv2d_42/Conv2D/ReadVariableOpReadVariableOpGsequential_9_module_wrapper_86_conv2d_42_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
/sequential_9/module_wrapper_86/conv2d_42/Conv2DConv2D;sequential_9/module_wrapper_85/conv2d_41/Relu:activations:0Fsequential_9/module_wrapper_86/conv2d_42/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
?
?sequential_9/module_wrapper_86/conv2d_42/BiasAdd/ReadVariableOpReadVariableOpHsequential_9_module_wrapper_86_conv2d_42_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
0sequential_9/module_wrapper_86/conv2d_42/BiasAddBiasAdd8sequential_9/module_wrapper_86/conv2d_42/Conv2D:output:0Gsequential_9/module_wrapper_86/conv2d_42/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@?
-sequential_9/module_wrapper_86/conv2d_42/ReluRelu9sequential_9/module_wrapper_86/conv2d_42/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@?
7sequential_9/module_wrapper_87/max_pooling2d_26/MaxPoolMaxPool;sequential_9/module_wrapper_86/conv2d_42/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
?
2sequential_9/batch_normalization_21/ReadVariableOpReadVariableOp;sequential_9_batch_normalization_21_readvariableop_resource*
_output_shapes
:@*
dtype0?
4sequential_9/batch_normalization_21/ReadVariableOp_1ReadVariableOp=sequential_9_batch_normalization_21_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
Csequential_9/batch_normalization_21/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_9_batch_normalization_21_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
Esequential_9/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_9_batch_normalization_21_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
4sequential_9/batch_normalization_21/FusedBatchNormV3FusedBatchNormV3@sequential_9/module_wrapper_87/max_pooling2d_26/MaxPool:output:0:sequential_9/batch_normalization_21/ReadVariableOp:value:0<sequential_9/batch_normalization_21/ReadVariableOp_1:value:0Ksequential_9/batch_normalization_21/FusedBatchNormV3/ReadVariableOp:value:0Msequential_9/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
is_training( ?
>sequential_9/module_wrapper_88/conv2d_43/Conv2D/ReadVariableOpReadVariableOpGsequential_9_module_wrapper_88_conv2d_43_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
/sequential_9/module_wrapper_88/conv2d_43/Conv2DConv2D8sequential_9/batch_normalization_21/FusedBatchNormV3:y:0Fsequential_9/module_wrapper_88/conv2d_43/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????

?*
paddingVALID*
strides
?
?sequential_9/module_wrapper_88/conv2d_43/BiasAdd/ReadVariableOpReadVariableOpHsequential_9_module_wrapper_88_conv2d_43_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
0sequential_9/module_wrapper_88/conv2d_43/BiasAddBiasAdd8sequential_9/module_wrapper_88/conv2d_43/Conv2D:output:0Gsequential_9/module_wrapper_88/conv2d_43/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????

??
-sequential_9/module_wrapper_88/conv2d_43/ReluRelu9sequential_9/module_wrapper_88/conv2d_43/BiasAdd:output:0*
T0*0
_output_shapes
:?????????

??
>sequential_9/module_wrapper_89/conv2d_44/Conv2D/ReadVariableOpReadVariableOpGsequential_9_module_wrapper_89_conv2d_44_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
/sequential_9/module_wrapper_89/conv2d_44/Conv2DConv2D;sequential_9/module_wrapper_88/conv2d_43/Relu:activations:0Fsequential_9/module_wrapper_89/conv2d_44/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
?
?sequential_9/module_wrapper_89/conv2d_44/BiasAdd/ReadVariableOpReadVariableOpHsequential_9_module_wrapper_89_conv2d_44_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
0sequential_9/module_wrapper_89/conv2d_44/BiasAddBiasAdd8sequential_9/module_wrapper_89/conv2d_44/Conv2D:output:0Gsequential_9/module_wrapper_89/conv2d_44/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:???????????
-sequential_9/module_wrapper_89/conv2d_44/ReluRelu9sequential_9/module_wrapper_89/conv2d_44/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
7sequential_9/module_wrapper_90/max_pooling2d_27/MaxPoolMaxPool;sequential_9/module_wrapper_89/conv2d_44/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
?
2sequential_9/batch_normalization_22/ReadVariableOpReadVariableOp;sequential_9_batch_normalization_22_readvariableop_resource*
_output_shapes	
:?*
dtype0?
4sequential_9/batch_normalization_22/ReadVariableOp_1ReadVariableOp=sequential_9_batch_normalization_22_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
Csequential_9/batch_normalization_22/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_9_batch_normalization_22_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Esequential_9/batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_9_batch_normalization_22_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
4sequential_9/batch_normalization_22/FusedBatchNormV3FusedBatchNormV3@sequential_9/module_wrapper_90/max_pooling2d_27/MaxPool:output:0:sequential_9/batch_normalization_22/ReadVariableOp:value:0<sequential_9/batch_normalization_22/ReadVariableOp_1:value:0Ksequential_9/batch_normalization_22/FusedBatchNormV3/ReadVariableOp:value:0Msequential_9/batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( ?
>sequential_9/module_wrapper_91/conv2d_45/Conv2D/ReadVariableOpReadVariableOpGsequential_9_module_wrapper_91_conv2d_45_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
/sequential_9/module_wrapper_91/conv2d_45/Conv2DConv2D8sequential_9/batch_normalization_22/FusedBatchNormV3:y:0Fsequential_9/module_wrapper_91/conv2d_45/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
?
?sequential_9/module_wrapper_91/conv2d_45/BiasAdd/ReadVariableOpReadVariableOpHsequential_9_module_wrapper_91_conv2d_45_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
0sequential_9/module_wrapper_91/conv2d_45/BiasAddBiasAdd8sequential_9/module_wrapper_91/conv2d_45/Conv2D:output:0Gsequential_9/module_wrapper_91/conv2d_45/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:???????????
-sequential_9/module_wrapper_91/conv2d_45/ReluRelu9sequential_9/module_wrapper_91/conv2d_45/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
7sequential_9/module_wrapper_92/max_pooling2d_28/MaxPoolMaxPool;sequential_9/module_wrapper_91/conv2d_45/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
?
2sequential_9/batch_normalization_23/ReadVariableOpReadVariableOp;sequential_9_batch_normalization_23_readvariableop_resource*
_output_shapes	
:?*
dtype0?
4sequential_9/batch_normalization_23/ReadVariableOp_1ReadVariableOp=sequential_9_batch_normalization_23_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
Csequential_9/batch_normalization_23/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_9_batch_normalization_23_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Esequential_9/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_9_batch_normalization_23_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
4sequential_9/batch_normalization_23/FusedBatchNormV3FusedBatchNormV3@sequential_9/module_wrapper_92/max_pooling2d_28/MaxPool:output:0:sequential_9/batch_normalization_23/ReadVariableOp:value:0<sequential_9/batch_normalization_23/ReadVariableOp_1:value:0Ksequential_9/batch_normalization_23/FusedBatchNormV3/ReadVariableOp:value:0Msequential_9/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 
.sequential_9/module_wrapper_93/flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ?
0sequential_9/module_wrapper_93/flatten_9/ReshapeReshape8sequential_9/batch_normalization_23/FusedBatchNormV3:y:07sequential_9/module_wrapper_93/flatten_9/Const:output:0*
T0*(
_output_shapes
:???????????
=sequential_9/module_wrapper_94/dense_18/MatMul/ReadVariableOpReadVariableOpFsequential_9_module_wrapper_94_dense_18_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
.sequential_9/module_wrapper_94/dense_18/MatMulMatMul9sequential_9/module_wrapper_93/flatten_9/Reshape:output:0Esequential_9/module_wrapper_94/dense_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
>sequential_9/module_wrapper_94/dense_18/BiasAdd/ReadVariableOpReadVariableOpGsequential_9_module_wrapper_94_dense_18_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
/sequential_9/module_wrapper_94/dense_18/BiasAddBiasAdd8sequential_9/module_wrapper_94/dense_18/MatMul:product:0Fsequential_9/module_wrapper_94/dense_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
,sequential_9/module_wrapper_94/dense_18/ReluRelu8sequential_9/module_wrapper_94/dense_18/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
=sequential_9/module_wrapper_95/dense_19/MatMul/ReadVariableOpReadVariableOpFsequential_9_module_wrapper_95_dense_19_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
.sequential_9/module_wrapper_95/dense_19/MatMulMatMul:sequential_9/module_wrapper_94/dense_18/Relu:activations:0Esequential_9/module_wrapper_95/dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
>sequential_9/module_wrapper_95/dense_19/BiasAdd/ReadVariableOpReadVariableOpGsequential_9_module_wrapper_95_dense_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
/sequential_9/module_wrapper_95/dense_19/BiasAddBiasAdd8sequential_9/module_wrapper_95/dense_19/MatMul:product:0Fsequential_9/module_wrapper_95/dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
/sequential_9/module_wrapper_95/dense_19/SoftmaxSoftmax8sequential_9/module_wrapper_95/dense_19/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
IdentityIdentity9sequential_9/module_wrapper_95/dense_19/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOpD^sequential_9/batch_normalization_21/FusedBatchNormV3/ReadVariableOpF^sequential_9/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_13^sequential_9/batch_normalization_21/ReadVariableOp5^sequential_9/batch_normalization_21/ReadVariableOp_1D^sequential_9/batch_normalization_22/FusedBatchNormV3/ReadVariableOpF^sequential_9/batch_normalization_22/FusedBatchNormV3/ReadVariableOp_13^sequential_9/batch_normalization_22/ReadVariableOp5^sequential_9/batch_normalization_22/ReadVariableOp_1D^sequential_9/batch_normalization_23/FusedBatchNormV3/ReadVariableOpF^sequential_9/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_13^sequential_9/batch_normalization_23/ReadVariableOp5^sequential_9/batch_normalization_23/ReadVariableOp_1@^sequential_9/module_wrapper_85/conv2d_41/BiasAdd/ReadVariableOp?^sequential_9/module_wrapper_85/conv2d_41/Conv2D/ReadVariableOp@^sequential_9/module_wrapper_86/conv2d_42/BiasAdd/ReadVariableOp?^sequential_9/module_wrapper_86/conv2d_42/Conv2D/ReadVariableOp@^sequential_9/module_wrapper_88/conv2d_43/BiasAdd/ReadVariableOp?^sequential_9/module_wrapper_88/conv2d_43/Conv2D/ReadVariableOp@^sequential_9/module_wrapper_89/conv2d_44/BiasAdd/ReadVariableOp?^sequential_9/module_wrapper_89/conv2d_44/Conv2D/ReadVariableOp@^sequential_9/module_wrapper_91/conv2d_45/BiasAdd/ReadVariableOp?^sequential_9/module_wrapper_91/conv2d_45/Conv2D/ReadVariableOp?^sequential_9/module_wrapper_94/dense_18/BiasAdd/ReadVariableOp>^sequential_9/module_wrapper_94/dense_18/MatMul/ReadVariableOp?^sequential_9/module_wrapper_95/dense_19/BiasAdd/ReadVariableOp>^sequential_9/module_wrapper_95/dense_19/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : 2?
Csequential_9/batch_normalization_21/FusedBatchNormV3/ReadVariableOpCsequential_9/batch_normalization_21/FusedBatchNormV3/ReadVariableOp2?
Esequential_9/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1Esequential_9/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_12h
2sequential_9/batch_normalization_21/ReadVariableOp2sequential_9/batch_normalization_21/ReadVariableOp2l
4sequential_9/batch_normalization_21/ReadVariableOp_14sequential_9/batch_normalization_21/ReadVariableOp_12?
Csequential_9/batch_normalization_22/FusedBatchNormV3/ReadVariableOpCsequential_9/batch_normalization_22/FusedBatchNormV3/ReadVariableOp2?
Esequential_9/batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1Esequential_9/batch_normalization_22/FusedBatchNormV3/ReadVariableOp_12h
2sequential_9/batch_normalization_22/ReadVariableOp2sequential_9/batch_normalization_22/ReadVariableOp2l
4sequential_9/batch_normalization_22/ReadVariableOp_14sequential_9/batch_normalization_22/ReadVariableOp_12?
Csequential_9/batch_normalization_23/FusedBatchNormV3/ReadVariableOpCsequential_9/batch_normalization_23/FusedBatchNormV3/ReadVariableOp2?
Esequential_9/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1Esequential_9/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_12h
2sequential_9/batch_normalization_23/ReadVariableOp2sequential_9/batch_normalization_23/ReadVariableOp2l
4sequential_9/batch_normalization_23/ReadVariableOp_14sequential_9/batch_normalization_23/ReadVariableOp_12?
?sequential_9/module_wrapper_85/conv2d_41/BiasAdd/ReadVariableOp?sequential_9/module_wrapper_85/conv2d_41/BiasAdd/ReadVariableOp2?
>sequential_9/module_wrapper_85/conv2d_41/Conv2D/ReadVariableOp>sequential_9/module_wrapper_85/conv2d_41/Conv2D/ReadVariableOp2?
?sequential_9/module_wrapper_86/conv2d_42/BiasAdd/ReadVariableOp?sequential_9/module_wrapper_86/conv2d_42/BiasAdd/ReadVariableOp2?
>sequential_9/module_wrapper_86/conv2d_42/Conv2D/ReadVariableOp>sequential_9/module_wrapper_86/conv2d_42/Conv2D/ReadVariableOp2?
?sequential_9/module_wrapper_88/conv2d_43/BiasAdd/ReadVariableOp?sequential_9/module_wrapper_88/conv2d_43/BiasAdd/ReadVariableOp2?
>sequential_9/module_wrapper_88/conv2d_43/Conv2D/ReadVariableOp>sequential_9/module_wrapper_88/conv2d_43/Conv2D/ReadVariableOp2?
?sequential_9/module_wrapper_89/conv2d_44/BiasAdd/ReadVariableOp?sequential_9/module_wrapper_89/conv2d_44/BiasAdd/ReadVariableOp2?
>sequential_9/module_wrapper_89/conv2d_44/Conv2D/ReadVariableOp>sequential_9/module_wrapper_89/conv2d_44/Conv2D/ReadVariableOp2?
?sequential_9/module_wrapper_91/conv2d_45/BiasAdd/ReadVariableOp?sequential_9/module_wrapper_91/conv2d_45/BiasAdd/ReadVariableOp2?
>sequential_9/module_wrapper_91/conv2d_45/Conv2D/ReadVariableOp>sequential_9/module_wrapper_91/conv2d_45/Conv2D/ReadVariableOp2?
>sequential_9/module_wrapper_94/dense_18/BiasAdd/ReadVariableOp>sequential_9/module_wrapper_94/dense_18/BiasAdd/ReadVariableOp2~
=sequential_9/module_wrapper_94/dense_18/MatMul/ReadVariableOp=sequential_9/module_wrapper_94/dense_18/MatMul/ReadVariableOp2?
>sequential_9/module_wrapper_95/dense_19/BiasAdd/ReadVariableOp>sequential_9/module_wrapper_95/dense_19/BiasAdd/ReadVariableOp2~
=sequential_9/module_wrapper_95/dense_19/MatMul/ReadVariableOp=sequential_9/module_wrapper_95/dense_19/MatMul/ReadVariableOp:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_5
?M
?
H__inference_sequential_9_layer_call_and_return_conditional_losses_177159
input_52
module_wrapper_85_177092:@&
module_wrapper_85_177094:@2
module_wrapper_86_177097:@@&
module_wrapper_86_177099:@+
batch_normalization_21_177103:@+
batch_normalization_21_177105:@+
batch_normalization_21_177107:@+
batch_normalization_21_177109:@3
module_wrapper_88_177112:@?'
module_wrapper_88_177114:	?4
module_wrapper_89_177117:??'
module_wrapper_89_177119:	?,
batch_normalization_22_177123:	?,
batch_normalization_22_177125:	?,
batch_normalization_22_177127:	?,
batch_normalization_22_177129:	?4
module_wrapper_91_177132:??'
module_wrapper_91_177134:	?,
batch_normalization_23_177138:	?,
batch_normalization_23_177140:	?,
batch_normalization_23_177142:	?,
batch_normalization_23_177144:	?,
module_wrapper_94_177148:
??'
module_wrapper_94_177150:	?+
module_wrapper_95_177153:	?&
module_wrapper_95_177155:
identity??.batch_normalization_21/StatefulPartitionedCall?.batch_normalization_22/StatefulPartitionedCall?.batch_normalization_23/StatefulPartitionedCall?)module_wrapper_85/StatefulPartitionedCall?)module_wrapper_86/StatefulPartitionedCall?)module_wrapper_88/StatefulPartitionedCall?)module_wrapper_89/StatefulPartitionedCall?)module_wrapper_91/StatefulPartitionedCall?)module_wrapper_94/StatefulPartitionedCall?)module_wrapper_95/StatefulPartitionedCall?
)module_wrapper_85/StatefulPartitionedCallStatefulPartitionedCallinput_5module_wrapper_85_177092module_wrapper_85_177094*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_85_layer_call_and_return_conditional_losses_176771?
)module_wrapper_86/StatefulPartitionedCallStatefulPartitionedCall2module_wrapper_85/StatefulPartitionedCall:output:0module_wrapper_86_177097module_wrapper_86_177099*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_86_layer_call_and_return_conditional_losses_176741?
!module_wrapper_87/PartitionedCallPartitionedCall2module_wrapper_86/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_87_layer_call_and_return_conditional_losses_176715?
.batch_normalization_21/StatefulPartitionedCallStatefulPartitionedCall*module_wrapper_87/PartitionedCall:output:0batch_normalization_21_177103batch_normalization_21_177105batch_normalization_21_177107batch_normalization_21_177109*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_176126?
)module_wrapper_88/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_21/StatefulPartitionedCall:output:0module_wrapper_88_177112module_wrapper_88_177114*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????

?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_88_layer_call_and_return_conditional_losses_176695?
)module_wrapper_89/StatefulPartitionedCallStatefulPartitionedCall2module_wrapper_88/StatefulPartitionedCall:output:0module_wrapper_89_177117module_wrapper_89_177119*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_89_layer_call_and_return_conditional_losses_176665?
!module_wrapper_90/PartitionedCallPartitionedCall2module_wrapper_89/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_90_layer_call_and_return_conditional_losses_176639?
.batch_normalization_22/StatefulPartitionedCallStatefulPartitionedCall*module_wrapper_90/PartitionedCall:output:0batch_normalization_22_177123batch_normalization_22_177125batch_normalization_22_177127batch_normalization_22_177129*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_176190?
)module_wrapper_91/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_22/StatefulPartitionedCall:output:0module_wrapper_91_177132module_wrapper_91_177134*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_91_layer_call_and_return_conditional_losses_176619?
!module_wrapper_92/PartitionedCallPartitionedCall2module_wrapper_91/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_92_layer_call_and_return_conditional_losses_176593?
.batch_normalization_23/StatefulPartitionedCallStatefulPartitionedCall*module_wrapper_92/PartitionedCall:output:0batch_normalization_23_177138batch_normalization_23_177140batch_normalization_23_177142batch_normalization_23_177144*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_176254?
!module_wrapper_93/PartitionedCallPartitionedCall7batch_normalization_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_93_layer_call_and_return_conditional_losses_176577?
)module_wrapper_94/StatefulPartitionedCallStatefulPartitionedCall*module_wrapper_93/PartitionedCall:output:0module_wrapper_94_177148module_wrapper_94_177150*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_94_layer_call_and_return_conditional_losses_176556?
)module_wrapper_95/StatefulPartitionedCallStatefulPartitionedCall2module_wrapper_94/StatefulPartitionedCall:output:0module_wrapper_95_177153module_wrapper_95_177155*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_95_layer_call_and_return_conditional_losses_176526?
IdentityIdentity2module_wrapper_95/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp/^batch_normalization_21/StatefulPartitionedCall/^batch_normalization_22/StatefulPartitionedCall/^batch_normalization_23/StatefulPartitionedCall*^module_wrapper_85/StatefulPartitionedCall*^module_wrapper_86/StatefulPartitionedCall*^module_wrapper_88/StatefulPartitionedCall*^module_wrapper_89/StatefulPartitionedCall*^module_wrapper_91/StatefulPartitionedCall*^module_wrapper_94/StatefulPartitionedCall*^module_wrapper_95/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_21/StatefulPartitionedCall.batch_normalization_21/StatefulPartitionedCall2`
.batch_normalization_22/StatefulPartitionedCall.batch_normalization_22/StatefulPartitionedCall2`
.batch_normalization_23/StatefulPartitionedCall.batch_normalization_23/StatefulPartitionedCall2V
)module_wrapper_85/StatefulPartitionedCall)module_wrapper_85/StatefulPartitionedCall2V
)module_wrapper_86/StatefulPartitionedCall)module_wrapper_86/StatefulPartitionedCall2V
)module_wrapper_88/StatefulPartitionedCall)module_wrapper_88/StatefulPartitionedCall2V
)module_wrapper_89/StatefulPartitionedCall)module_wrapper_89/StatefulPartitionedCall2V
)module_wrapper_91/StatefulPartitionedCall)module_wrapper_91/StatefulPartitionedCall2V
)module_wrapper_94/StatefulPartitionedCall)module_wrapper_94/StatefulPartitionedCall2V
)module_wrapper_95/StatefulPartitionedCall)module_wrapper_95/StatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_5"?
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
C
input_58
serving_default_input_5:0?????????E
module_wrapper_950
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer-6
layer_with_weights-5
layer-7
	layer_with_weights-6
	layer-8

layer-9
layer_with_weights-7
layer-10
layer-11
layer_with_weights-8
layer-12
layer_with_weights-9
layer-13
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_module"
_tf_keras_layer
?
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses
%_module"
_tf_keras_layer
?
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses
,_module"
_tf_keras_layer
?
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses
3axis
	4gamma
5beta
6moving_mean
7moving_variance"
_tf_keras_layer
?
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses
>_module"
_tf_keras_layer
?
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses
E_module"
_tf_keras_layer
?
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J__call__
*K&call_and_return_all_conditional_losses
L_module"
_tf_keras_layer
?
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses
Saxis
	Tgamma
Ubeta
Vmoving_mean
Wmoving_variance"
_tf_keras_layer
?
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses
^_module"
_tf_keras_layer
?
_	variables
`trainable_variables
aregularization_losses
b	keras_api
c__call__
*d&call_and_return_all_conditional_losses
e_module"
_tf_keras_layer
?
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
j__call__
*k&call_and_return_all_conditional_losses
laxis
	mgamma
nbeta
omoving_mean
pmoving_variance"
_tf_keras_layer
?
q	variables
rtrainable_variables
sregularization_losses
t	keras_api
u__call__
*v&call_and_return_all_conditional_losses
w_module"
_tf_keras_layer
?
x	variables
ytrainable_variables
zregularization_losses
{	keras_api
|__call__
*}&call_and_return_all_conditional_losses
~_module"
_tf_keras_layer
?
	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_module"
_tf_keras_layer
?
?0
?1
?2
?3
44
55
66
77
?8
?9
?10
?11
T12
U13
V14
W15
?16
?17
m18
n19
o20
p21
?22
?23
?24
?25"
trackable_list_wrapper
?
?0
?1
?2
?3
44
55
?6
?7
?8
?9
T10
U11
?12
?13
m14
n15
?16
?17
?18
?19"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_1
?trace_2
?trace_32?
-__inference_sequential_9_layer_call_fn_176503
-__inference_sequential_9_layer_call_fn_177277
-__inference_sequential_9_layer_call_fn_177334
-__inference_sequential_9_layer_call_fn_177019?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1z?trace_2z?trace_3
?
?trace_0
?trace_1
?trace_2
?trace_32?
H__inference_sequential_9_layer_call_and_return_conditional_losses_177434
H__inference_sequential_9_layer_call_and_return_conditional_losses_177534
H__inference_sequential_9_layer_call_and_return_conditional_losses_177089
H__inference_sequential_9_layer_call_and_return_conditional_losses_177159?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1z?trace_2z?trace_3
?B?
!__inference__wrapped_model_176073input_5"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?
?
_variables
?_iterations
?_learning_rate
?_index_dict
?
_momentums
?_velocities
?_update_step_xla"
experimentalOptimizer
-
?serving_default"
signature_map
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
2__inference_module_wrapper_85_layer_call_fn_177543
2__inference_module_wrapper_85_layer_call_fn_177552?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
M__inference_module_wrapper_85_layer_call_and_return_conditional_losses_177563
M__inference_module_wrapper_85_layer_call_and_return_conditional_losses_177574?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 z?trace_0z?trace_1
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias"
_tf_keras_layer
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
2__inference_module_wrapper_86_layer_call_fn_177584
2__inference_module_wrapper_86_layer_call_fn_177593?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
M__inference_module_wrapper_86_layer_call_and_return_conditional_losses_177604
M__inference_module_wrapper_86_layer_call_and_return_conditional_losses_177615?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 z?trace_0z?trace_1
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
2__inference_module_wrapper_87_layer_call_fn_177620
2__inference_module_wrapper_87_layer_call_fn_177625?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
M__inference_module_wrapper_87_layer_call_and_return_conditional_losses_177630
M__inference_module_wrapper_87_layer_call_and_return_conditional_losses_177635?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 z?trace_0z?trace_1
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
<
40
51
62
73"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
7__inference_batch_normalization_21_layer_call_fn_177660
7__inference_batch_normalization_21_layer_call_fn_177673?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_177691
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_177709?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1
 "
trackable_list_wrapper
*:(@2batch_normalization_21/gamma
):'@2batch_normalization_21/beta
2:0@ (2"batch_normalization_21/moving_mean
6:4@ (2&batch_normalization_21/moving_variance
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
2__inference_module_wrapper_88_layer_call_fn_177718
2__inference_module_wrapper_88_layer_call_fn_177727?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
M__inference_module_wrapper_88_layer_call_and_return_conditional_losses_177738
M__inference_module_wrapper_88_layer_call_and_return_conditional_losses_177749?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 z?trace_0z?trace_1
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias"
_tf_keras_layer
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
2__inference_module_wrapper_89_layer_call_fn_177758
2__inference_module_wrapper_89_layer_call_fn_177767?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
M__inference_module_wrapper_89_layer_call_and_return_conditional_losses_177778
M__inference_module_wrapper_89_layer_call_and_return_conditional_losses_177789?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 z?trace_0z?trace_1
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
2__inference_module_wrapper_90_layer_call_fn_177794
2__inference_module_wrapper_90_layer_call_fn_177799?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
M__inference_module_wrapper_90_layer_call_and_return_conditional_losses_177804
M__inference_module_wrapper_90_layer_call_and_return_conditional_losses_177809?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 z?trace_0z?trace_1
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
<
T0
U1
V2
W3"
trackable_list_wrapper
.
T0
U1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
7__inference_batch_normalization_22_layer_call_fn_177834
7__inference_batch_normalization_22_layer_call_fn_177847?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_177865
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_177883?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1
 "
trackable_list_wrapper
+:)?2batch_normalization_22/gamma
*:(?2batch_normalization_22/beta
3:1? (2"batch_normalization_22/moving_mean
7:5? (2&batch_normalization_22/moving_variance
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
2__inference_module_wrapper_91_layer_call_fn_177892
2__inference_module_wrapper_91_layer_call_fn_177901?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
M__inference_module_wrapper_91_layer_call_and_return_conditional_losses_177912
M__inference_module_wrapper_91_layer_call_and_return_conditional_losses_177923?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 z?trace_0z?trace_1
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
_	variables
`trainable_variables
aregularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
2__inference_module_wrapper_92_layer_call_fn_177928
2__inference_module_wrapper_92_layer_call_fn_177933?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
M__inference_module_wrapper_92_layer_call_and_return_conditional_losses_177938
M__inference_module_wrapper_92_layer_call_and_return_conditional_losses_177943?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 z?trace_0z?trace_1
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
<
m0
n1
o2
p3"
trackable_list_wrapper
.
m0
n1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
f	variables
gtrainable_variables
hregularization_losses
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
7__inference_batch_normalization_23_layer_call_fn_177968
7__inference_batch_normalization_23_layer_call_fn_177981?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_177999
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_178017?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1
 "
trackable_list_wrapper
+:)?2batch_normalization_23/gamma
*:(?2batch_normalization_23/beta
3:1? (2"batch_normalization_23/moving_mean
7:5? (2&batch_normalization_23/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
q	variables
rtrainable_variables
sregularization_losses
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
2__inference_module_wrapper_93_layer_call_fn_178022
2__inference_module_wrapper_93_layer_call_fn_178027?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
M__inference_module_wrapper_93_layer_call_and_return_conditional_losses_178033
M__inference_module_wrapper_93_layer_call_and_return_conditional_losses_178039?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 z?trace_0z?trace_1
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
x	variables
ytrainable_variables
zregularization_losses
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
2__inference_module_wrapper_94_layer_call_fn_178048
2__inference_module_wrapper_94_layer_call_fn_178057?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
M__inference_module_wrapper_94_layer_call_and_return_conditional_losses_178068
M__inference_module_wrapper_94_layer_call_and_return_conditional_losses_178079?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 z?trace_0z?trace_1
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias"
_tf_keras_layer
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
2__inference_module_wrapper_95_layer_call_fn_178088
2__inference_module_wrapper_95_layer_call_fn_178097?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
M__inference_module_wrapper_95_layer_call_and_return_conditional_losses_178108
M__inference_module_wrapper_95_layer_call_and_return_conditional_losses_178119?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 z?trace_0z?trace_1
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias"
_tf_keras_layer
<::@2"module_wrapper_85/conv2d_41/kernel
.:,@2 module_wrapper_85/conv2d_41/bias
<::@@2"module_wrapper_86/conv2d_42/kernel
.:,@2 module_wrapper_86/conv2d_42/bias
=:;@?2"module_wrapper_88/conv2d_43/kernel
/:-?2 module_wrapper_88/conv2d_43/bias
>:<??2"module_wrapper_89/conv2d_44/kernel
/:-?2 module_wrapper_89/conv2d_44/bias
>:<??2"module_wrapper_91/conv2d_45/kernel
/:-?2 module_wrapper_91/conv2d_45/bias
5:3
??2!module_wrapper_94/dense_18/kernel
.:,?2module_wrapper_94/dense_18/bias
4:2	?2!module_wrapper_95/dense_19/kernel
-:+2module_wrapper_95/dense_19/bias
J
60
71
V2
W3
o4
p5"
trackable_list_wrapper
?
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
13"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
-__inference_sequential_9_layer_call_fn_176503input_5"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
-__inference_sequential_9_layer_call_fn_177277inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
-__inference_sequential_9_layer_call_fn_177334inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
-__inference_sequential_9_layer_call_fn_177019input_5"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
H__inference_sequential_9_layer_call_and_return_conditional_losses_177434inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
H__inference_sequential_9_layer_call_and_return_conditional_losses_177534inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
H__inference_sequential_9_layer_call_and_return_conditional_losses_177089input_5"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
H__inference_sequential_9_layer_call_and_return_conditional_losses_177159input_5"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?
?0
?1
?2
?3
?4
?5
?6
?7
?8
?9
?10
?11
?12
?13
?14
?15
?16
?17
?18
?19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35
?36
?37
?38
?39
?40"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
?
?0
?1
?2
?3
?4
?5
?6
?7
?8
?9
?10
?11
?12
?13
?14
?15
?16
?17
?18
?19"
trackable_list_wrapper
?
?0
?1
?2
?3
?4
?5
?6
?7
?8
?9
?10
?11
?12
?13
?14
?15
?16
?17
?18
?19"
trackable_list_wrapper
?2??
???
FullArgSpec2
args*?'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
?B?
$__inference_signature_wrapper_177220input_5"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
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
?B?
2__inference_module_wrapper_85_layer_call_fn_177543args_0"?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
2__inference_module_wrapper_85_layer_call_fn_177552args_0"?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
M__inference_module_wrapper_85_layer_call_and_return_conditional_losses_177563args_0"?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
M__inference_module_wrapper_85_layer_call_and_return_conditional_losses_177574args_0"?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?metrics
?	variables
?non_trainable_variables
?layer_metrics
?regularization_losses
?trainable_variables
?layers
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
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
?B?
2__inference_module_wrapper_86_layer_call_fn_177584args_0"?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
2__inference_module_wrapper_86_layer_call_fn_177593args_0"?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
M__inference_module_wrapper_86_layer_call_and_return_conditional_losses_177604args_0"?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
M__inference_module_wrapper_86_layer_call_and_return_conditional_losses_177615args_0"?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?metrics
?	variables
?non_trainable_variables
?layer_metrics
?regularization_losses
?trainable_variables
?layers
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
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
?B?
2__inference_module_wrapper_87_layer_call_fn_177620args_0"?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
2__inference_module_wrapper_87_layer_call_fn_177625args_0"?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
M__inference_module_wrapper_87_layer_call_and_return_conditional_losses_177630args_0"?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
M__inference_module_wrapper_87_layer_call_and_return_conditional_losses_177635args_0"?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?	variables
?non_trainable_variables
?layer_metrics
?regularization_losses
?trainable_variables
?layers
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
1__inference_max_pooling2d_26_layer_call_fn_177647?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????z?trace_0
?
?trace_02?
L__inference_max_pooling2d_26_layer_call_and_return_conditional_losses_177641?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????z?trace_0
.
60
71"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
7__inference_batch_normalization_21_layer_call_fn_177660inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
7__inference_batch_normalization_21_layer_call_fn_177673inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_177691inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_177709inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
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
?B?
2__inference_module_wrapper_88_layer_call_fn_177718args_0"?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
2__inference_module_wrapper_88_layer_call_fn_177727args_0"?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
M__inference_module_wrapper_88_layer_call_and_return_conditional_losses_177738args_0"?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
M__inference_module_wrapper_88_layer_call_and_return_conditional_losses_177749args_0"?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?metrics
?	variables
?non_trainable_variables
?layer_metrics
?regularization_losses
?trainable_variables
?layers
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
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
?B?
2__inference_module_wrapper_89_layer_call_fn_177758args_0"?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
2__inference_module_wrapper_89_layer_call_fn_177767args_0"?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
M__inference_module_wrapper_89_layer_call_and_return_conditional_losses_177778args_0"?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
M__inference_module_wrapper_89_layer_call_and_return_conditional_losses_177789args_0"?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?metrics
?	variables
?non_trainable_variables
?layer_metrics
?regularization_losses
?trainable_variables
?layers
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
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
?B?
2__inference_module_wrapper_90_layer_call_fn_177794args_0"?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
2__inference_module_wrapper_90_layer_call_fn_177799args_0"?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
M__inference_module_wrapper_90_layer_call_and_return_conditional_losses_177804args_0"?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
M__inference_module_wrapper_90_layer_call_and_return_conditional_losses_177809args_0"?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?	variables
?non_trainable_variables
?layer_metrics
?regularization_losses
?trainable_variables
?layers
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
1__inference_max_pooling2d_27_layer_call_fn_177821?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????z?trace_0
?
?trace_02?
L__inference_max_pooling2d_27_layer_call_and_return_conditional_losses_177815?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????z?trace_0
.
V0
W1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
7__inference_batch_normalization_22_layer_call_fn_177834inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
7__inference_batch_normalization_22_layer_call_fn_177847inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_177865inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_177883inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
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
?B?
2__inference_module_wrapper_91_layer_call_fn_177892args_0"?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
2__inference_module_wrapper_91_layer_call_fn_177901args_0"?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
M__inference_module_wrapper_91_layer_call_and_return_conditional_losses_177912args_0"?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
M__inference_module_wrapper_91_layer_call_and_return_conditional_losses_177923args_0"?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?metrics
?	variables
?non_trainable_variables
?layer_metrics
?regularization_losses
?trainable_variables
?layers
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
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
?B?
2__inference_module_wrapper_92_layer_call_fn_177928args_0"?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
2__inference_module_wrapper_92_layer_call_fn_177933args_0"?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
M__inference_module_wrapper_92_layer_call_and_return_conditional_losses_177938args_0"?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
M__inference_module_wrapper_92_layer_call_and_return_conditional_losses_177943args_0"?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?	variables
?non_trainable_variables
?layer_metrics
?regularization_losses
?trainable_variables
?layers
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
1__inference_max_pooling2d_28_layer_call_fn_177955?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????z?trace_0
?
?trace_02?
L__inference_max_pooling2d_28_layer_call_and_return_conditional_losses_177949?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????z?trace_0
.
o0
p1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
7__inference_batch_normalization_23_layer_call_fn_177968inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
7__inference_batch_normalization_23_layer_call_fn_177981inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_177999inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_178017inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
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
?B?
2__inference_module_wrapper_93_layer_call_fn_178022args_0"?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
2__inference_module_wrapper_93_layer_call_fn_178027args_0"?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
M__inference_module_wrapper_93_layer_call_and_return_conditional_losses_178033args_0"?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
M__inference_module_wrapper_93_layer_call_and_return_conditional_losses_178039args_0"?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?	variables
?non_trainable_variables
?layer_metrics
?regularization_losses
?trainable_variables
?layers
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
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
?B?
2__inference_module_wrapper_94_layer_call_fn_178048args_0"?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
2__inference_module_wrapper_94_layer_call_fn_178057args_0"?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
M__inference_module_wrapper_94_layer_call_and_return_conditional_losses_178068args_0"?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
M__inference_module_wrapper_94_layer_call_and_return_conditional_losses_178079args_0"?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?metrics
?	variables
?non_trainable_variables
?layer_metrics
?regularization_losses
?trainable_variables
?layers
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
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
?B?
2__inference_module_wrapper_95_layer_call_fn_178088args_0"?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
2__inference_module_wrapper_95_layer_call_fn_178097args_0"?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
M__inference_module_wrapper_95_layer_call_and_return_conditional_losses_178108args_0"?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
M__inference_module_wrapper_95_layer_call_and_return_conditional_losses_178119args_0"?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?metrics
?	variables
?non_trainable_variables
?layer_metrics
?regularization_losses
?trainable_variables
?layers
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
R
?	variables
?	keras_api

?total

?count"
_tf_keras_metric
c
?	variables
?	keras_api

?total

?count
?
_fn_kwargs"
_tf_keras_metric
A:?@2)Adam/m/module_wrapper_85/conv2d_41/kernel
A:?@2)Adam/v/module_wrapper_85/conv2d_41/kernel
3:1@2'Adam/m/module_wrapper_85/conv2d_41/bias
3:1@2'Adam/v/module_wrapper_85/conv2d_41/bias
A:?@@2)Adam/m/module_wrapper_86/conv2d_42/kernel
A:?@@2)Adam/v/module_wrapper_86/conv2d_42/kernel
3:1@2'Adam/m/module_wrapper_86/conv2d_42/bias
3:1@2'Adam/v/module_wrapper_86/conv2d_42/bias
/:-@2#Adam/m/batch_normalization_21/gamma
/:-@2#Adam/v/batch_normalization_21/gamma
.:,@2"Adam/m/batch_normalization_21/beta
.:,@2"Adam/v/batch_normalization_21/beta
B:@@?2)Adam/m/module_wrapper_88/conv2d_43/kernel
B:@@?2)Adam/v/module_wrapper_88/conv2d_43/kernel
4:2?2'Adam/m/module_wrapper_88/conv2d_43/bias
4:2?2'Adam/v/module_wrapper_88/conv2d_43/bias
C:A??2)Adam/m/module_wrapper_89/conv2d_44/kernel
C:A??2)Adam/v/module_wrapper_89/conv2d_44/kernel
4:2?2'Adam/m/module_wrapper_89/conv2d_44/bias
4:2?2'Adam/v/module_wrapper_89/conv2d_44/bias
0:.?2#Adam/m/batch_normalization_22/gamma
0:.?2#Adam/v/batch_normalization_22/gamma
/:-?2"Adam/m/batch_normalization_22/beta
/:-?2"Adam/v/batch_normalization_22/beta
C:A??2)Adam/m/module_wrapper_91/conv2d_45/kernel
C:A??2)Adam/v/module_wrapper_91/conv2d_45/kernel
4:2?2'Adam/m/module_wrapper_91/conv2d_45/bias
4:2?2'Adam/v/module_wrapper_91/conv2d_45/bias
0:.?2#Adam/m/batch_normalization_23/gamma
0:.?2#Adam/v/batch_normalization_23/gamma
/:-?2"Adam/m/batch_normalization_23/beta
/:-?2"Adam/v/batch_normalization_23/beta
::8
??2(Adam/m/module_wrapper_94/dense_18/kernel
::8
??2(Adam/v/module_wrapper_94/dense_18/kernel
3:1?2&Adam/m/module_wrapper_94/dense_18/bias
3:1?2&Adam/v/module_wrapper_94/dense_18/bias
9:7	?2(Adam/m/module_wrapper_95/dense_19/kernel
9:7	?2(Adam/v/module_wrapper_95/dense_19/kernel
2:02&Adam/m/module_wrapper_95/dense_19/bias
2:02&Adam/v/module_wrapper_95/dense_19/bias
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
?B?
1__inference_max_pooling2d_26_layer_call_fn_177647inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?B?
L__inference_max_pooling2d_26_layer_call_and_return_conditional_losses_177641inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
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
?B?
1__inference_max_pooling2d_27_layer_call_fn_177821inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?B?
L__inference_max_pooling2d_27_layer_call_and_return_conditional_losses_177815inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
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
?B?
1__inference_max_pooling2d_28_layer_call_fn_177955inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?B?
L__inference_max_pooling2d_28_layer_call_and_return_conditional_losses_177949inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
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
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper?
!__inference__wrapped_model_176073?(????4567????TUVW??mnop????8?5
.?+
)?&
input_5?????????
? "E?B
@
module_wrapper_95+?(
module_wrapper_95??????????
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_177691?4567M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "F?C
<?9
tensor_0+???????????????????????????@
? ?
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_177709?4567M?J
C?@
:?7
inputs+???????????????????????????@
p
? "F?C
<?9
tensor_0+???????????????????????????@
? ?
7__inference_batch_normalization_21_layer_call_fn_177660?4567M?J
C?@
:?7
inputs+???????????????????????????@
p 
? ";?8
unknown+???????????????????????????@?
7__inference_batch_normalization_21_layer_call_fn_177673?4567M?J
C?@
:?7
inputs+???????????????????????????@
p
? ";?8
unknown+???????????????????????????@?
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_177865?TUVWN?K
D?A
;?8
inputs,????????????????????????????
p 
? "G?D
=?:
tensor_0,????????????????????????????
? ?
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_177883?TUVWN?K
D?A
;?8
inputs,????????????????????????????
p
? "G?D
=?:
tensor_0,????????????????????????????
? ?
7__inference_batch_normalization_22_layer_call_fn_177834?TUVWN?K
D?A
;?8
inputs,????????????????????????????
p 
? "<?9
unknown,?????????????????????????????
7__inference_batch_normalization_22_layer_call_fn_177847?TUVWN?K
D?A
;?8
inputs,????????????????????????????
p
? "<?9
unknown,?????????????????????????????
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_177999?mnopN?K
D?A
;?8
inputs,????????????????????????????
p 
? "G?D
=?:
tensor_0,????????????????????????????
? ?
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_178017?mnopN?K
D?A
;?8
inputs,????????????????????????????
p
? "G?D
=?:
tensor_0,????????????????????????????
? ?
7__inference_batch_normalization_23_layer_call_fn_177968?mnopN?K
D?A
;?8
inputs,????????????????????????????
p 
? "<?9
unknown,?????????????????????????????
7__inference_batch_normalization_23_layer_call_fn_177981?mnopN?K
D?A
;?8
inputs,????????????????????????????
p
? "<?9
unknown,?????????????????????????????
L__inference_max_pooling2d_26_layer_call_and_return_conditional_losses_177641?R?O
H?E
C?@
inputs4????????????????????????????????????
? "O?L
E?B
tensor_04????????????????????????????????????
? ?
1__inference_max_pooling2d_26_layer_call_fn_177647?R?O
H?E
C?@
inputs4????????????????????????????????????
? "D?A
unknown4?????????????????????????????????????
L__inference_max_pooling2d_27_layer_call_and_return_conditional_losses_177815?R?O
H?E
C?@
inputs4????????????????????????????????????
? "O?L
E?B
tensor_04????????????????????????????????????
? ?
1__inference_max_pooling2d_27_layer_call_fn_177821?R?O
H?E
C?@
inputs4????????????????????????????????????
? "D?A
unknown4?????????????????????????????????????
L__inference_max_pooling2d_28_layer_call_and_return_conditional_losses_177949?R?O
H?E
C?@
inputs4????????????????????????????????????
? "O?L
E?B
tensor_04????????????????????????????????????
? ?
1__inference_max_pooling2d_28_layer_call_fn_177955?R?O
H?E
C?@
inputs4????????????????????????????????????
? "D?A
unknown4?????????????????????????????????????
M__inference_module_wrapper_85_layer_call_and_return_conditional_losses_177563???G?D
-?*
(?%
args_0?????????
?

trainingp "4?1
*?'
tensor_0?????????@
? ?
M__inference_module_wrapper_85_layer_call_and_return_conditional_losses_177574???G?D
-?*
(?%
args_0?????????
?

trainingp"4?1
*?'
tensor_0?????????@
? ?
2__inference_module_wrapper_85_layer_call_fn_177543z??G?D
-?*
(?%
args_0?????????
?

trainingp ")?&
unknown?????????@?
2__inference_module_wrapper_85_layer_call_fn_177552z??G?D
-?*
(?%
args_0?????????
?

trainingp")?&
unknown?????????@?
M__inference_module_wrapper_86_layer_call_and_return_conditional_losses_177604???G?D
-?*
(?%
args_0?????????@
?

trainingp "4?1
*?'
tensor_0?????????@
? ?
M__inference_module_wrapper_86_layer_call_and_return_conditional_losses_177615???G?D
-?*
(?%
args_0?????????@
?

trainingp"4?1
*?'
tensor_0?????????@
? ?
2__inference_module_wrapper_86_layer_call_fn_177584z??G?D
-?*
(?%
args_0?????????@
?

trainingp ")?&
unknown?????????@?
2__inference_module_wrapper_86_layer_call_fn_177593z??G?D
-?*
(?%
args_0?????????@
?

trainingp")?&
unknown?????????@?
M__inference_module_wrapper_87_layer_call_and_return_conditional_losses_177630G?D
-?*
(?%
args_0?????????@
?

trainingp "4?1
*?'
tensor_0?????????@
? ?
M__inference_module_wrapper_87_layer_call_and_return_conditional_losses_177635G?D
-?*
(?%
args_0?????????@
?

trainingp"4?1
*?'
tensor_0?????????@
? ?
2__inference_module_wrapper_87_layer_call_fn_177620tG?D
-?*
(?%
args_0?????????@
?

trainingp ")?&
unknown?????????@?
2__inference_module_wrapper_87_layer_call_fn_177625tG?D
-?*
(?%
args_0?????????@
?

trainingp")?&
unknown?????????@?
M__inference_module_wrapper_88_layer_call_and_return_conditional_losses_177738???G?D
-?*
(?%
args_0?????????@
?

trainingp "5?2
+?(
tensor_0?????????

?
? ?
M__inference_module_wrapper_88_layer_call_and_return_conditional_losses_177749???G?D
-?*
(?%
args_0?????????@
?

trainingp"5?2
+?(
tensor_0?????????

?
? ?
2__inference_module_wrapper_88_layer_call_fn_177718{??G?D
-?*
(?%
args_0?????????@
?

trainingp "*?'
unknown?????????

??
2__inference_module_wrapper_88_layer_call_fn_177727{??G?D
-?*
(?%
args_0?????????@
?

trainingp"*?'
unknown?????????

??
M__inference_module_wrapper_89_layer_call_and_return_conditional_losses_177778???H?E
.?+
)?&
args_0?????????

?
?

trainingp "5?2
+?(
tensor_0??????????
? ?
M__inference_module_wrapper_89_layer_call_and_return_conditional_losses_177789???H?E
.?+
)?&
args_0?????????

?
?

trainingp"5?2
+?(
tensor_0??????????
? ?
2__inference_module_wrapper_89_layer_call_fn_177758|??H?E
.?+
)?&
args_0?????????

?
?

trainingp "*?'
unknown???????????
2__inference_module_wrapper_89_layer_call_fn_177767|??H?E
.?+
)?&
args_0?????????

?
?

trainingp"*?'
unknown???????????
M__inference_module_wrapper_90_layer_call_and_return_conditional_losses_177804?H?E
.?+
)?&
args_0??????????
?

trainingp "5?2
+?(
tensor_0??????????
? ?
M__inference_module_wrapper_90_layer_call_and_return_conditional_losses_177809?H?E
.?+
)?&
args_0??????????
?

trainingp"5?2
+?(
tensor_0??????????
? ?
2__inference_module_wrapper_90_layer_call_fn_177794vH?E
.?+
)?&
args_0??????????
?

trainingp "*?'
unknown???????????
2__inference_module_wrapper_90_layer_call_fn_177799vH?E
.?+
)?&
args_0??????????
?

trainingp"*?'
unknown???????????
M__inference_module_wrapper_91_layer_call_and_return_conditional_losses_177912???H?E
.?+
)?&
args_0??????????
?

trainingp "5?2
+?(
tensor_0??????????
? ?
M__inference_module_wrapper_91_layer_call_and_return_conditional_losses_177923???H?E
.?+
)?&
args_0??????????
?

trainingp"5?2
+?(
tensor_0??????????
? ?
2__inference_module_wrapper_91_layer_call_fn_177892|??H?E
.?+
)?&
args_0??????????
?

trainingp "*?'
unknown???????????
2__inference_module_wrapper_91_layer_call_fn_177901|??H?E
.?+
)?&
args_0??????????
?

trainingp"*?'
unknown???????????
M__inference_module_wrapper_92_layer_call_and_return_conditional_losses_177938?H?E
.?+
)?&
args_0??????????
?

trainingp "5?2
+?(
tensor_0??????????
? ?
M__inference_module_wrapper_92_layer_call_and_return_conditional_losses_177943?H?E
.?+
)?&
args_0??????????
?

trainingp"5?2
+?(
tensor_0??????????
? ?
2__inference_module_wrapper_92_layer_call_fn_177928vH?E
.?+
)?&
args_0??????????
?

trainingp "*?'
unknown???????????
2__inference_module_wrapper_92_layer_call_fn_177933vH?E
.?+
)?&
args_0??????????
?

trainingp"*?'
unknown???????????
M__inference_module_wrapper_93_layer_call_and_return_conditional_losses_178033yH?E
.?+
)?&
args_0??????????
?

trainingp "-?*
#? 
tensor_0??????????
? ?
M__inference_module_wrapper_93_layer_call_and_return_conditional_losses_178039yH?E
.?+
)?&
args_0??????????
?

trainingp"-?*
#? 
tensor_0??????????
? ?
2__inference_module_wrapper_93_layer_call_fn_178022nH?E
.?+
)?&
args_0??????????
?

trainingp ""?
unknown???????????
2__inference_module_wrapper_93_layer_call_fn_178027nH?E
.?+
)?&
args_0??????????
?

trainingp""?
unknown???????????
M__inference_module_wrapper_94_layer_call_and_return_conditional_losses_178068w??@?=
&?#
!?
args_0??????????
?

trainingp "-?*
#? 
tensor_0??????????
? ?
M__inference_module_wrapper_94_layer_call_and_return_conditional_losses_178079w??@?=
&?#
!?
args_0??????????
?

trainingp"-?*
#? 
tensor_0??????????
? ?
2__inference_module_wrapper_94_layer_call_fn_178048l??@?=
&?#
!?
args_0??????????
?

trainingp ""?
unknown???????????
2__inference_module_wrapper_94_layer_call_fn_178057l??@?=
&?#
!?
args_0??????????
?

trainingp""?
unknown???????????
M__inference_module_wrapper_95_layer_call_and_return_conditional_losses_178108v??@?=
&?#
!?
args_0??????????
?

trainingp ",?)
"?
tensor_0?????????
? ?
M__inference_module_wrapper_95_layer_call_and_return_conditional_losses_178119v??@?=
&?#
!?
args_0??????????
?

trainingp",?)
"?
tensor_0?????????
? ?
2__inference_module_wrapper_95_layer_call_fn_178088k??@?=
&?#
!?
args_0??????????
?

trainingp "!?
unknown??????????
2__inference_module_wrapper_95_layer_call_fn_178097k??@?=
&?#
!?
args_0??????????
?

trainingp"!?
unknown??????????
H__inference_sequential_9_layer_call_and_return_conditional_losses_177089?(????4567????TUVW??mnop????@?=
6?3
)?&
input_5?????????
p 

 
? ",?)
"?
tensor_0?????????
? ?
H__inference_sequential_9_layer_call_and_return_conditional_losses_177159?(????4567????TUVW??mnop????@?=
6?3
)?&
input_5?????????
p

 
? ",?)
"?
tensor_0?????????
? ?
H__inference_sequential_9_layer_call_and_return_conditional_losses_177434?(????4567????TUVW??mnop??????<
5?2
(?%
inputs?????????
p 

 
? ",?)
"?
tensor_0?????????
? ?
H__inference_sequential_9_layer_call_and_return_conditional_losses_177534?(????4567????TUVW??mnop??????<
5?2
(?%
inputs?????????
p

 
? ",?)
"?
tensor_0?????????
? ?
-__inference_sequential_9_layer_call_fn_176503?(????4567????TUVW??mnop????@?=
6?3
)?&
input_5?????????
p 

 
? "!?
unknown??????????
-__inference_sequential_9_layer_call_fn_177019?(????4567????TUVW??mnop????@?=
6?3
)?&
input_5?????????
p

 
? "!?
unknown??????????
-__inference_sequential_9_layer_call_fn_177277?(????4567????TUVW??mnop??????<
5?2
(?%
inputs?????????
p 

 
? "!?
unknown??????????
-__inference_sequential_9_layer_call_fn_177334?(????4567????TUVW??mnop??????<
5?2
(?%
inputs?????????
p

 
? "!?
unknown??????????
$__inference_signature_wrapper_177220?(????4567????TUVW??mnop????C?@
? 
9?6
4
input_5)?&
input_5?????????"E?B
@
module_wrapper_95+?(
module_wrapper_95?????????
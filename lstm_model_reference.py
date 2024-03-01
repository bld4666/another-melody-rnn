class LSTMCell(Layer, DropoutRNNCell):
    def __init__(
        self,
        units,
        activation="tanh",
        recurrent_activation="sigmoid",
        use_bias=True,
        kernel_initializer="glorot_uniform",
        recurrent_initializer="orthogonal",
        bias_initializer="zeros",
        unit_forget_bias=True,
        kernel_regularizer=None,
        recurrent_regularizer=None,
        bias_regularizer=None,
        kernel_constraint=None,
        recurrent_constraint=None,
        bias_constraint=None,
        dropout=0.0,
        recurrent_dropout=0.0,
        seed=None,
        **kwargs,
    ):
        if units <= 0:
            raise ValueError(
                "Received an invalid value for argument `units`, "
                f"expected a positive integer, got {units}."
            )
        super().__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1.0, max(0.0, dropout))
        self.recurrent_dropout = min(1.0, max(0.0, recurrent_dropout))
        self.seed = seed
        self.seed_generator = backend.random.SeedGenerator(seed=seed)

        self.unit_forget_bias = unit_forget_bias
        self.state_size = [self.units, self.units]
        self.output_size = self.units

    def build(self, input_shape):
        super().build(input_shape)
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(
            shape=(input_dim, self.units * 4),
            name="kernel",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 4),
            name="recurrent_kernel",
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
        )

        if self.use_bias:
            if self.unit_forget_bias:

                def bias_initializer(_, *args, **kwargs):
                    return ops.concatenate(
                        [
                            self.bias_initializer(
                                (self.units,), *args, **kwargs
                            ),
                            initializers.get("ones")(
                                (self.units,), *args, **kwargs
                            ),
                            self.bias_initializer(
                                (self.units * 2,), *args, **kwargs
                            ),
                        ]
                    )

            else:
                bias_initializer = self.bias_initializer
            self.bias = self.add_weight(
                shape=(self.units * 4,),
                name="bias",
                initializer=bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.bias = None
        self.built = True

    def _compute_carry_and_output(self, x, h_tm1, c_tm1):
        x_i, x_f, x_c, x_o = x
        h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o = h_tm1
        i = self.recurrent_activation(
            x_i + ops.matmul(h_tm1_i, self.recurrent_kernel[:, : self.units])
        )
        f = self.recurrent_activation(
            x_f
            + ops.matmul(
                h_tm1_f, self.recurrent_kernel[:, self.units : self.units * 2]
            )
        )
        c = f * c_tm1 + i * self.activation(
            x_c
            + ops.matmul(
                h_tm1_c,
                self.recurrent_kernel[:, self.units * 2 : self.units * 3],
            )
        )
        o = self.recurrent_activation(
            x_o
            + ops.matmul(h_tm1_o, self.recurrent_kernel[:, self.units * 3 :])
        )
        return c, o

    def call(self, inputs, states, training=False):
        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state

        dp_mask = self.get_dropout_mask(inputs)
        rec_dp_mask = self.get_recurrent_dropout_mask(h_tm1)

        if training and 0.0 < self.dropout < 1.0:
            inputs = inputs * dp_mask
        if training and 0.0 < self.recurrent_dropout < 1.0:
            h_tm1 = h_tm1 * rec_dp_mask

        inputs_i = inputs
        inputs_f = inputs
        inputs_c = inputs
        inputs_o = inputs
        k_i, k_f, k_c, k_o = ops.split(self.kernel, 4, axis=1)
        x_i = ops.matmul(inputs_i, k_i)
        x_f = ops.matmul(inputs_f, k_f)
        x_c = ops.matmul(inputs_c, k_c)
        x_o = ops.matmul(inputs_o, k_o)
        if self.use_bias:
            b_i, b_f, b_c, b_o = ops.split(self.bias, 4, axis=0)
            x_i += b_i
            x_f += b_f
            x_c += b_c
            x_o += b_o

        h_tm1_i = h_tm1
        h_tm1_f = h_tm1
        h_tm1_c = h_tm1
        h_tm1_o = h_tm1
        x = (x_i, x_f, x_c, x_o)
        h_tm1 = (h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o)
        c, o = self._compute_carry_and_output(x, h_tm1, c_tm1)
        h = o * self.activation(c)

        return h, [h, c]

    def get_config(self):
        config = {
            "units": self.units,
            "activation": activations.serialize(self.activation),
            "recurrent_activation": activations.serialize(
                self.recurrent_activation
            ),
            "use_bias": self.use_bias,
            "unit_forget_bias": self.unit_forget_bias,
            "kernel_initializer": initializers.serialize(
                self.kernel_initializer
            ),
            "recurrent_initializer": initializers.serialize(
                self.recurrent_initializer
            ),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(
                self.kernel_regularizer
            ),
            "recurrent_regularizer": regularizers.serialize(
                self.recurrent_regularizer
            ),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "recurrent_constraint": constraints.serialize(
                self.recurrent_constraint
            ),
            "bias_constraint": constraints.serialize(self.bias_constraint),
            "dropout": self.dropout,
            "recurrent_dropout": self.recurrent_dropout,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def get_initial_state(self, batch_size=None):
        return [
            ops.zeros((batch_size, d), dtype=self.compute_dtype)
            for d in self.state_size
        ]


@keras_export("keras.layers.LSTM")
class LSTM(RNN):
    def __init__(
        self,
        units,
        activation="tanh",
        recurrent_activation="sigmoid",
        use_bias=True,
        kernel_initializer="glorot_uniform",
        recurrent_initializer="orthogonal",
        bias_initializer="zeros",
        unit_forget_bias=True,
        kernel_regularizer=None,
        recurrent_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        recurrent_constraint=None,
        bias_constraint=None,
        dropout=0.0,
        recurrent_dropout=0.0,
        seed=None,
        return_sequences=False,
        return_state=False,
        go_backwards=False,
        stateful=False,
        unroll=False,
        **kwargs,
    ):
        cell = LSTMCell(
            units,
            activation=activation,
            recurrent_activation=recurrent_activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            unit_forget_bias=unit_forget_bias,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            dtype=kwargs.get("dtype", None),
            trainable=kwargs.get("trainable", True),
            name="lstm_cell",
            seed=seed,
        )
        super().__init__(
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            unroll=unroll,
            activity_regularizer=activity_regularizer,
            **kwargs,
        )
        self.input_spec = InputSpec(ndim=3)
        if backend.backend() == "tensorflow" and backend.cudnn_ok(
            cell.activation,
            cell.recurrent_activation,
            self.unroll,
            cell.use_bias,
        ):
            self.supports_jit = False

    def inner_loop(self, sequences, initial_state, mask, training=False):
        if tree.is_nested(mask):
            mask = mask[0]

        if not self.dropout and not self.recurrent_dropout:
            try:
                out = backend.lstm(
                    sequences,
                    initial_state[0],
                    initial_state[1],
                    mask,
                    kernel=self.cell.kernel,
                    recurrent_kernel=self.cell.recurrent_kernel,
                    bias=self.cell.bias,
                    activation=self.cell.activation,
                    recurrent_activation=self.cell.recurrent_activation,
                    return_sequences=self.return_sequences,
                    go_backwards=self.go_backwards,
                    unroll=self.unroll,
                )
                if backend.backend() == "tensorflow":
                    self.supports_jit = False
                return out
            except NotImplementedError:
                pass
        return super().inner_loop(
            sequences, initial_state, mask=mask, training=training
        )

    def call(self, sequences, initial_state=None, mask=None, training=False):
        timesteps = sequences.shape[1]
        if self.unroll and timesteps is None:
            raise ValueError(
                "Cannot unroll a RNN if the "
                "time dimension is undefined. \n"
                "- If using a Sequential model, "
                "specify the time dimension by passing "
                "an `Input()` as your first layer.\n"
                "- If using the functional API, specify "
                "the time dimension by passing a `shape` "
                "or `batch_shape` argument to your `Input()`."
            )

        if initial_state is None:
            if self.stateful:
                initial_state = self.states
            else:
                initial_state = self.get_initial_state(
                    batch_size=ops.shape(sequences)[0]
                )
        # RNN expect the states in a list, even if single state.
        if not tree.is_nested(initial_state):
            initial_state = [initial_state]
        initial_state = list(initial_state)

        # Cast states to compute dtype.
        # Note that states may be deeply nested
        # (e.g. in the stacked cells case).
        initial_state = tree.map_structure(
            lambda x: backend.convert_to_tensor(
                x, dtype=self.cell.compute_dtype
            ),
            initial_state,
        )

        # Prepopulate the dropout state so that the inner_loop is stateless
        # this is particularly important for JAX backend.
        self._maybe_config_dropout_masks(self.cell, sequences, initial_state)

        last_output, outputs, states = self.inner_loop(
            sequences=sequences,
            initial_state=initial_state,
            mask=mask,
            training=training,
        )
        last_output = ops.cast(last_output, self.compute_dtype)
        outputs = ops.cast(outputs, self.compute_dtype)
        states = tree.map_structure(
            lambda x: ops.cast(x, dtype=self.compute_dtype), states
        )
        self._maybe_reset_dropout_masks(self.cell)

        if self.stateful:
            for self_state, state in zip(
                tree.flatten(self.states), tree.flatten(states)
            ):
                self_state.assign(state)

        if self.return_sequences:
            output = outputs
        else:
            output = last_output

        if self.return_state:
            if len(states) == 1:
                state = states[0]
                return output, state
            return output, *states
        return output
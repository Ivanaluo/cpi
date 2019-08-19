

    def forward(self,  src_seq, src_len):

    	output = self.embedding(src_seq)
    	max_len = torch.max(src_len)
    	input_pos = torch.LongTensor([list(range(1, len + 1)) + [0] * (max_len - len) for len in src_len])
    	pe_output = self.position_encoding(input_pos)
    	output += pe_output
    	key, value, query = output, output, output
    	residual = query
    	key, value, query  = self.linear_k(key), self.linear_v(value), self.linear_q(query)
    	key = key.view(batch_size * num_heads, -1, dim_per_head)
		value = value.view(batch_size * num_heads, -1, dim_per_head)
		query = query.view(batch_size * num_heads, -1, dim_per_head)
		self_attention_mask = padding_mask(src_seq, src_seq)
		# attn_mask = self_attention_mask 
		if attn_max_lenmask:
			attn_mask = attn_mask.repeat(num_heads, 1, 1)
		scale = (key.size(-1) // num_heads) ** -0.5  
		k, v, q = key, value, query
		attention = torch.bmm(q, k.transpose(1, 2))  
		if scale:
		    attention = attention * scale
		if attn_mask:
		    attention = attention.masked_fill_(attn_mask, -np.inf)
		attention = self.softmax(attention)
		attention = self.attn_dropout(attention)
		context = torch.bmm(attention, v)
		context = context.view(batch_size, -1, dim_per_head * num_heads) 
		output = self.linear_final(context)
		output = dropout(0.2)
		output = layer_norm(residual + output)
		context, attention = output, attention
		output = self.feed_forward(context)
		output, enc_self_attn = output, attention
		pooled = torch.cat((output.mean(1), torch.max(output, dim = 1)), dim = 1)

        return pooled, enc_self_attn





































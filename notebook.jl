### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ 25e36fc6-7b76-11eb-02df-bd87dde5cdfb
begin
	using Flux: Chain, Dense, params, logitcrossentropy, onehotbatch, ADAM, train!, softmax
	using Test
end

# ╔═╡ 8fe7bc4c-7b85-11eb-0287-0f179e325305
md"### Data Generation"

# ╔═╡ fde731a6-7b84-11eb-3758-23db3b0d2057
function fizzbuzz(x::Int)
    is_divisible_by_three = x % 3 == 0
    is_divisible_by_five = x % 5 == 0

    if is_divisible_by_three & is_divisible_by_five
        return "fizzbuzz"
    elseif is_divisible_by_three
        return "fizz"
    elseif is_divisible_by_five
        return "buzz"
    else
        return "else"
    end
end

# ╔═╡ 3a290946-7b85-11eb-39d9-87db6c6beef1
const LABELS = ["fizz", "buzz", "fizzbuzz", "else"];

# ╔═╡ 7f695632-7b85-11eb-1804-a9876b2c027f
md"### Feature Engineering"

# ╔═╡ 735140d0-7b85-11eb-0aaf-e3b29e66fa00
features(x) = float.([x % 3, x % 5, x % 15])

# ╔═╡ 7351a052-7b85-11eb-3a5d-e323871eb7ad
features(x::AbstractArray) = hcat(features.(x)...)

# ╔═╡ a0765410-7b85-11eb-076e-6f877fff0c95
features(1:15)

# ╔═╡ efff94ba-7b85-11eb-0d9e-d333585c6ed3
md"### Data Preparation"

# ╔═╡ f9de33f6-7b85-11eb-3eb0-6fcf7a369345
function getdata()
    
    @test fizzbuzz.([3, 5, 15, 98]) == LABELS
    
    raw_x = 1:100;
    raw_y = fizzbuzz.(raw_x);
    
    X = features(raw_x);
    y = onehotbatch(raw_y, LABELS);
    return X, y
end

# ╔═╡ 05d0994c-7b86-11eb-09ff-a33769bc2fc2
raw_x = 1:100

# ╔═╡ 73acb7d4-7b86-11eb-2666-b91efed75c66
raw_y = fizzbuzz.(raw_x)

# ╔═╡ 73ad3c68-7b86-11eb-27a3-815a9fa17317
X = features(raw_x)

# ╔═╡ 73b0a77c-7b86-11eb-15ae-fbb1afdb09c3
y = onehotbatch(raw_y, LABELS)

# ╔═╡ a45bc3b6-7b86-11eb-2b77-c5ce1e85d93c
m = Chain(Dense(3, 10), Dense(10, 4))

# ╔═╡ 3515921a-7b87-11eb-3639-f1369417430e
l1, l2 = m.layers

# ╔═╡ 4ae19f9c-7b87-11eb-326d-f36d3f129e86
m(features(3))

# ╔═╡ aadac77c-7b87-11eb-1720-07dad016249d
loss(x, y) = logitcrossentropy(m(x), y)

# ╔═╡ ba3420a6-7b87-11eb-33d9-4bf0d6ac359e
loss(X, y)

# ╔═╡ e2a834f0-7b87-11eb-10db-7151b60248ae
opt = ADAM()

# ╔═╡ e3e37032-7b87-11eb-3eff-c7592e3878ce
train!(loss, params(m), [(X, y)], opt)

# ╔═╡ 627c3cf8-7b88-11eb-0992-0fffc92c84ca
for e in 1:500
	train!(loss, params(m), [(X, y)], opt)
end

# ╔═╡ 103c356a-7b88-11eb-0c1b-a37efe27b9ef
deepbuzz(x) = (a = argmax(m(features(x))); a == 4 ? x : LABELS[a])

# ╔═╡ 346d4b2c-7b88-11eb-3713-ef621dc50ef5
deepbuzz.(1:15)

# ╔═╡ 5e38196e-7b88-11eb-3997-b52bcd52d76d


# ╔═╡ Cell order:
# ╠═25e36fc6-7b76-11eb-02df-bd87dde5cdfb
# ╟─8fe7bc4c-7b85-11eb-0287-0f179e325305
# ╠═fde731a6-7b84-11eb-3758-23db3b0d2057
# ╠═3a290946-7b85-11eb-39d9-87db6c6beef1
# ╟─7f695632-7b85-11eb-1804-a9876b2c027f
# ╠═735140d0-7b85-11eb-0aaf-e3b29e66fa00
# ╠═7351a052-7b85-11eb-3a5d-e323871eb7ad
# ╠═a0765410-7b85-11eb-076e-6f877fff0c95
# ╟─efff94ba-7b85-11eb-0d9e-d333585c6ed3
# ╠═f9de33f6-7b85-11eb-3eb0-6fcf7a369345
# ╠═05d0994c-7b86-11eb-09ff-a33769bc2fc2
# ╠═73acb7d4-7b86-11eb-2666-b91efed75c66
# ╠═73ad3c68-7b86-11eb-27a3-815a9fa17317
# ╠═73b0a77c-7b86-11eb-15ae-fbb1afdb09c3
# ╠═a45bc3b6-7b86-11eb-2b77-c5ce1e85d93c
# ╠═3515921a-7b87-11eb-3639-f1369417430e
# ╠═4ae19f9c-7b87-11eb-326d-f36d3f129e86
# ╠═aadac77c-7b87-11eb-1720-07dad016249d
# ╠═ba3420a6-7b87-11eb-33d9-4bf0d6ac359e
# ╠═e2a834f0-7b87-11eb-10db-7151b60248ae
# ╠═e3e37032-7b87-11eb-3eff-c7592e3878ce
# ╠═627c3cf8-7b88-11eb-0992-0fffc92c84ca
# ╠═103c356a-7b88-11eb-0c1b-a37efe27b9ef
# ╠═346d4b2c-7b88-11eb-3713-ef621dc50ef5
# ╠═5e38196e-7b88-11eb-3997-b52bcd52d76d

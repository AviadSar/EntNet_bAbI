import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext.data import get_tokenizer
from data_utils import load_task

n_input_words = 20
embedding_dim = 100
n_memories = 20
gradient_clip_value = 40
tasks = [1]
data_dir = "data/tasks_1-20_v1-2/en-10k"


def get_data_tokens():
    tokenizer = get_tokenizer("basic_english")
    return set(tokenizer("this is a test sentence, it's purpose is to test the ability of my code to embed words"))


def init_embedding_matrix(vocab):
    token_to_idx = {token: i+1 for i, token in enumerate(vocab)}
    embeddings_matrix = nn.Embedding(len(vocab) + 1, embedding_dim, 0)

    return embeddings_matrix, token_to_idx

    # use the following 3 lines to obtain word embeddings. padding a tensor with 0 will pad it's embeddings with 0's vectors
    # tokens_to_embed = ["sentence", "purpose"]
    # tokens_to_embed_idx = torch.tensor([token_to_idx[w] for w in tokens_to_embed], dtype=torch.long)
    # embeddings = embeddings_matrix(tokens_to_embed_idx)


def vectorize_data(data, token_to_idx, embeddings_matrix):
    vec_data = []
    for story, query, answer in data:
        vec_story = torch.zeros(len(story), n_input_words, embedding_dim)
        for i, sentence in enumerate(story):
            ls = max(0, n_input_words - len(sentence))
            vec_story[i] = embeddings_matrix(torch.tensor([token_to_idx[w] for w in sentence] + [0] * ls))

        lq = max(0, n_input_words - len(query))
        vec_query = embeddings_matrix(torch.tensor([token_to_idx[w] for w in query] + [0] * lq))

        # vec_answer = torch.zeros(len(token_to_idx) + 1)  # 0 is reserved for nil word
        # for a in answer:
        #     vec_answer[token_to_idx[a]] = 1

        vec_answer = []  # 0 is reserved for nil word
        for a in answer:
            vec_answer.append(token_to_idx[a])
        vec_answer = torch.tensor(vec_answer)

        vec_data.append((vec_story, vec_query, vec_answer))

    return vec_data


def get_vocab(train, test):
    vocab = set()
    samples = train + test
    for story, query, answer in samples:
        for word in [word for sentence in story for word in sentence] + query:
            vocab.add(word)
    return vocab, len(vocab) + 1


def get_key_tensors(vocab, embeddings_matrix, token_to_idx, tied=True):
    """
    returns a list of key tensors with length n_memories
    list may be randomly initialized (current version) or tied to specific entities
    """
    if tied:
        keys = torch.zeros((n_memories, embedding_dim))
        for i, word in enumerate(vocab):
            if i < n_memories:
                keys[i] = embeddings_matrix(torch.tensor(token_to_idx[word]))
        return keys

    mean = torch.zeros((n_memories, embedding_dim))
    standard_deviation = torch.full((n_memories, embedding_dim), 0.1)
    return torch.normal(mean, standard_deviation)


def get_matrix_weights():
    """
    :return: initial weights for any og the matrices U, V, W
     weights may be randomly initialized (current version) or initialized to zeros or the identity matrix
    """
    init_mean = torch.zeros((embedding_dim, embedding_dim))
    init_standard_deviation = torch.full((embedding_dim, embedding_dim), 0.1)

    return nn.Parameter(torch.normal(init_mean, init_standard_deviation))


def get_r_matrix_weights(vocab_size):
    """
    :return: initial weights for any og the matrices U, V, W
     weights may be randomly initialized (current version) or initialized to zeros or the identity matrix
    """
    init_mean = torch.zeros((vocab_size, embedding_dim))
    init_standard_deviation = torch.full((vocab_size, embedding_dim), 0.1)

    return nn.Parameter(torch.normal(init_mean, init_standard_deviation))


def get_non_linearity():
    """
    :return: the non-linearity function to be used in the model.
    this may be a parametric ReLU (current version) or (despite its name) the identity
    """
    return nn.PReLU(init=1)


##### Build Network #####
class EntNet(nn.Module):
    def __init__(self, vocab_size, keys):
        super(EntNet, self).__init__()
        # Encoder
        self.encoder_multiplier = nn.Parameter(torch.ones((n_input_words, embedding_dim)))

        # Memory
        self.keys = nn.Parameter(keys)
        self.memories = self.init_new_memories()

        # self.gates = nn.Parameter(torch.zeros(n_memories), requires_grad=True)

        self.U = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.V = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.W = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.U.weight = get_matrix_weights()
        self.V.weight = get_matrix_weights()
        self.V.weight = get_matrix_weights()

        self.non_linearity = get_non_linearity()

        # Decoder
        self.R = nn.Linear(vocab_size, embedding_dim, bias=False)
        self.H = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.R.weight = get_r_matrix_weights(vocab_size)
        self.H.weight = get_matrix_weights()

    def forward(self, input):
        # re-initialize memories to key-values
        self.memories = self.init_new_memories()
        # Encoder
        input = input * self.encoder_multiplier  # use torch.mul() if this fails
        input = input.sum(dim=1)  # or is it dim=0 ? TBC

        # Memory
        for i in range(n_memories):
            gate = F.sigmoid(torch.matmul(input, self.memories[i]) + torch.matmul(input, self.keys[i]))
            update_candidate = self.non_linearity(self.U(self.memories[i]) + self.V(self.keys[i]) + self.W(input))
            self.memories[i] = self.memories[i] + (update_candidate.t() * gate).t()
            self.memories[i] = self.memories[i] / torch.norm(self.memories[i])

    def decode(self, query):
        # Decoder
        query = query * self.encoder_multiplier  # use torch.mul() if this fails
        query = query.sum(dim=0)  # or is it dim=0 ? TBC
        answers_probabilities = F.softmax(torch.tensor([torch.matmul(query, self.memories[i].t()) for i in range(n_memories)]))
        score = sum([answers_probabilities[i] * self.memories[i] for i in range(n_memories)]) # TODO: make sure python sum works as expected in this case
        result = self.R(self.non_linearity(query + self.H(score)))
        return result

    def init_new_memories(self):
        memories = []
        for tensor in self.keys:
            memories.append(torch.tensor(tensor, requires_grad=False))
        return memories


def learn(task):
    train, test = load_task(data_dir, task)

    vocab, vocab_size = get_vocab(train, test)
    embeddings_matrix, token_to_idx = init_embedding_matrix(vocab)
    keys = get_key_tensors(vocab, embeddings_matrix, token_to_idx, True)

    vec_train = vectorize_data(train, token_to_idx, embeddings_matrix)
    vec_test = vectorize_data(test, token_to_idx, embeddings_matrix)

    entnet = EntNet(vocab_size, keys)

    ##### Define Loss and Optimizer #####
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(entnet.parameters(), lr=0.01)

    ##### Train Model #####
    epoch = 0
    prev_loss = None
    loss = None
    stuck_epochs = 0
    max_stuck_epochs = 3
    epsilon = 0.1
    while True:  # when to stop adding epochs?
        running_loss = 0.0
        for i, sample in enumerate(vec_train):
            # get the inputs; data is a list of [inputs, labels]
            story, query, answer = sample

            # zero the parameter gradients
            optimizer.zero_grad()

            for sentence in story:
                entnet(sentence.view(1, n_input_words, embedding_dim))
            output = entnet.decode(query)
            loss = criterion(output, answer)
            loss.backward()
            nn.utils.clip_grad_value_(entnet.parameters(), gradient_clip_value)
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 50 == 49:  # print every 50 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 50))
                running_loss = 0.0

            # correct = 0
            # pred_idx = np.argmax(output.detach().numpy())
            # # print("pred is: " + str(pred_idx) + ", answer is: " + str(answer[0].item()))
            # if pred_idx == answer[0].item():
            #     correct += 1
            # if i % 50 == 49:  # print every 50 mini-batches
            #     print('[%d, %5d] correct: %.3f' %
            #           (epoch + 1, i + 1, correct / 50))
            #     correct = 0

        if epoch == 0:
            prev_loss = loss
        elif prev_loss - loss < epsilon:
            stuck_epochs += 1
            prev_loss = loss

        if stuck_epochs > max_stuck_epochs:
            break

        # adjust learning rate every 25 epochs until 200 epochs
        if epoch < 200 and epoch % 25 == 24:
            optimizer.lr = optimizer.lr / 2
        epoch += 1
    print('Finished Training')


def main():
    for task in tasks:
        learn(task)


if __name__ == "__main__":
    main()
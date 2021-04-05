.. pyrtd documentation master file, created by
   sphinx-quickstart on Mon Aug 26 13:30:29 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Fully Homomorphic Encryption
############################

Fully Homomorphic Encryption (FHE) is the holy-grail of encryption, and the cypherpunks dream. FHE encrypted cyphertexts can be computed on/ used for processing in arbitrary computational-depth calculations while being continuously in cyphertext form, or rather without the ability of decrypting the cyphertexts by the data processor. What this means is it is now possible to process any such encrypted data without any possibility of data leaks, and without any ability to decrypt or discern a users data. Furthermore the answers or predictions given by these FHE compatible data processors is simply a transformation of the input cyphertext, meaning only the original encryptor of the data/ private key holder can decrypt these answers.

Data processors can process FHE cyphertexts, and can own their deep/ machine learning models as a service.

Users (including other industries) can keep their data, and predictions/ calculations completely private, in quantum-decryption resistant form without needing to give anyone else their private key. This includes highly sensitive domains such as diagnosis requiring very personal patient data which is now indecipherable.

Win-Win. This benefits all directly involved. However there are some drawbacks:

- processing cyphertexts is naturally more (computationally, spatially, and thus monetarily) expensive and slow. Orders of magnitude.
- processing cyphertexts is more complex, requiring that all operations are abelian compatible I.E addition, multiplication, addition of a negative, but not any division or true subtraction.
- FHE is still relatively new, and each implementation is slightly different and could still hold bugs. Most FHE implementations are not FHE yet as they have not implemented bootstrapping thus having a set maximum computational depth.

This library sets out to enable, and to make simple FHE. We want to see a future where users need not even disclose any personal information while still enabling meaningful, and impactfull predictions to be made for them. We envision this to play a deeply connected role with deep-learning in particular, enabling privacy-preserving automation; automotive, home, industrial. Privacy-preserving Diagnosis; medical, agricultural. Privacy-preserving services; entertainment, recommendation. This however will almost certainly be opposed, as it means a radical change to the status-quo, and often to the modus operandi of in particular advertisers (or data sellers); who are the groups who make the most money from personal information. Bad actors; who would seek to "break" this encryption to use the contents for nefarious or state sponsored purposes.
As you can probably see this adventure is going to be quite a difficult journey, we hope you bear with us
